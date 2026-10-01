import hashlib
import json
import multiprocessing
import os
import re
import subprocess
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import xxhash
from datasets import Dataset, Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer

ROOT = Path('/work/nat1andotxyz/lab4/dedup-s')
TOKENIZER_DIR = Path('/work/nat1andotxyz/lab4/c4-packed/tokenizer')
BLOCKS = 1_228_800
LENGTH = 1024
SEED = 42
BANDS = 14
HASHES_PER_BAND = 8
SIMILARITY = 0.8
METHODS = ('control', 'exact', 'minhash')


def initialize_worker():
    global tokenizer, coefficients, offsets
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    rng = np.random.default_rng(SEED)
    coefficients = rng.integers(1, 2**32, size=BANDS * HASHES_PER_BAND, dtype=np.uint64)
    offsets = rng.integers(0, 2**32, size=BANDS * HASHES_PER_BAND, dtype=np.uint64)


def process_batch(texts):
    tokenized = tokenizer(texts, add_special_tokens=False, truncation=False,
                          return_attention_mask=False, verbose=False)['input_ids']
    rows = []
    for text, tokens in zip(texts, tokenized):
        normalized = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
        digest = xxhash.xxh3_128_digest(normalized.encode())
        words = re.findall(r'\w+', normalized)
        signature = None
        if len(words) >= 5:
            shingles = np.unique(np.fromiter(
                (xxhash.xxh32_intdigest(' '.join(words[i:i + 5]).encode())
                 for i in range(len(words) - 4)), dtype=np.uint64))
            signature = np.full(BANDS * HASHES_PER_BAND, 2**32, dtype=np.uint64)
            for start in range(0, len(shingles), 512):
                values = (shingles[start:start + 512, None] * coefficients + offsets) % 4_294_967_311
                signature = np.minimum(signature, values.min(axis=0))
            signature = signature.astype(np.uint64)
        rows.append((tokens, digest, signature))
    return rows


def text_batches(documents):
    for batch in documents.iter(batch_size=32):
        yield batch['text']


class PackedWriter:
    def __init__(self, method):
        self.method = method
        self.path = ROOT / f'{method}.arrow'
        self.schema = Features({'input_ids': Sequence(Value('int32'), length=LENGTH)}).arrow_schema
        self.sink = pa.OSFile(str(self.path), 'wb')
        self.writer = pa.ipc.new_stream(self.sink, self.schema)
        self.pending = []
        self.buffer = []
        self.blocks = 0
        self.documents = 0
        self.rejected = 0
        self.tokens = 0

    @property
    def full(self):
        return self.blocks == BLOCKS

    def append(self, tokens, eos):
        if self.full:
            return
        self.documents += 1
        self.tokens += len(tokens) + 1
        self.pending.extend(tokens)
        self.pending.append(eos)
        end = min(len(self.pending) // LENGTH, BLOCKS - self.blocks) * LENGTH
        for start in range(0, end, LENGTH):
            self.buffer.append(self.pending[start:start + LENGTH])
            self.blocks += 1
            if len(self.buffer) == 1024:
                self.flush()
        self.pending = self.pending[end:] if not self.full else []

    def flush(self):
        if self.buffer:
            values = pa.array(np.asarray(self.buffer, dtype=np.int32).reshape(-1))
            array = pa.FixedSizeListArray.from_arrays(values, LENGTH)
            self.writer.write_batch(pa.RecordBatch.from_arrays([array], schema=self.schema))
            self.buffer.clear()

    def stats(self):
        return dict(blocks=self.blocks, accepted_documents=self.documents,
                    rejected_documents=self.rejected, accepted_tokens=self.tokens)

    def finish(self, metadata, tokenizer):
        self.flush()
        self.writer.close()
        self.sink.close()
        dataset = Dataset.from_file(str(self.path))
        if len(dataset) != BLOCKS:
            raise RuntimeError(f'{self.method}: only {len(dataset)} blocks')
        destination = ROOT / self.method
        staging = ROOT / f'{self.method}.building'
        dataset.save_to_disk(str(staging))
        tokenizer.save_pretrained(staging / 'tokenizer')
        details = dict(metadata, method=self.method, statistics=self.stats(),
                       dataset_fingerprint=dataset._fingerprint)
        (staging / 'dedup.json').write_text(json.dumps(details, indent=2))
        staging.rename(destination)
        self.path.unlink()


def prepare():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Data preparation requires a Slurm allocation')
    ROOT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    workers = max(1, int(os.environ['SLURM_CPUS_PER_TASK']) - 2)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    writers = {method: PackedWriter(method) for method in METHODS}
    exact_seen = set()
    near_seen = set()
    band_indices = [{} for _ in range(BANDS)]
    signatures = []
    scanned = 0
    stream_hash = hashlib.sha256()
    documents = load_dataset('allenai/c4', 'en', split='train', streaming=True)
    documents = documents.shuffle(seed=SEED, buffer_size=10_000)
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker) as pool:
        results = pool.map(process_batch, text_batches(documents), buffersize=workers * 2)
        for batch_index, rows in enumerate(results):
            for tokens, digest, signature in rows:
                scanned += 1
                stream_hash.update(digest)
                writers['control'].append(tokens, tokenizer.eos_token_id)
                exact = writers['exact']
                if not exact.full:
                    if digest in exact_seen:
                        exact.rejected += 1
                    else:
                        exact_seen.add(digest)
                        exact.append(tokens, tokenizer.eos_token_id)
                near = writers['minhash']
                if not near.full:
                    duplicate = digest in near_seen
                    keys = []
                    if signature is not None and not duplicate:
                        keys = [signature[i * HASHES_PER_BAND:(i + 1) * HASHES_PER_BAND].tobytes()
                                for i in range(BANDS)]
                        candidates = set()
                        for index, key in zip(band_indices, keys):
                            candidates.update(index.get(key, ()))
                        duplicate = any(np.mean(signature == signatures[candidate]) >= SIMILARITY
                                        for candidate in candidates)
                    if duplicate:
                        near.rejected += 1
                    else:
                        near_seen.add(digest)
                        if signature is not None:
                            position = len(signatures)
                            signatures.append(signature)
                            for index, key in zip(band_indices, keys):
                                index.setdefault(key, []).append(position)
                        near.append(tokens, tokenizer.eos_token_id)
                if all(writer.full for writer in writers.values()):
                    break
            if batch_index % 256 == 0:
                progress = dict(scanned_documents=scanned, elapsed_seconds=time.monotonic() - started,
                                methods={name: writer.stats() for name, writer in writers.items()})
                (ROOT / 'progress.json').write_text(json.dumps(progress, indent=2))
                print(json.dumps(progress), flush=True)
            if all(writer.full for writer in writers.values()):
                break
    if not all(writer.full for writer in writers.values()):
        raise RuntimeError('Source exhausted before all token budgets were filled')
    metadata = dict(dataset='allenai/c4', subset='en', split='train', seed=SEED,
                    shuffle_buffer=10_000, blocks=BLOCKS, sequence_length=LENGTH,
                    normalization='NFKC + casefold + whitespace collapse; original text is tokenized',
                    exact_hash='xxh3_128', shingle_words=5, minhash_hashes=BANDS * HASHES_PER_BAND,
                    minhash_bands=BANDS, minhash_hashes_per_band=HASHES_PER_BAND,
                    minhash_estimated_jaccard_threshold=SIMILARITY,
                    near_algorithm='greedy keep-first LSH candidates, estimated MinHash similarity',
                    scanned_documents=scanned, candidate_stream_sha256=stream_hash.hexdigest(),
                    preparation_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    preparation_job=os.environ['SLURM_JOB_ID'])
    for writer in writers.values():
        writer.finish(metadata, tokenizer)
    metadata['elapsed_seconds'] = time.monotonic() - started
    metadata['methods'] = {name: writer.stats() for name, writer in writers.items()}
    (ROOT / 'ready.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    prepare()
