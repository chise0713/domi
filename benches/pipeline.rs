use std::{
    env, fs,
    hint::black_box,
    path::{Path, PathBuf},
};

use criterion::{Criterion, criterion_group, criterion_main};
use domi::Entries;

fn load_dataset(path: &Path) -> Vec<(String, String)> {
    let mut files = Vec::new();

    for e in path.read_dir().unwrap() {
        let e = e.unwrap();

        if e.file_type().unwrap().is_file() {
            let name = e.file_name().into_string().unwrap();
            let content = fs::read_to_string(path.join(&name)).unwrap();

            files.push((name, content));
        }
    }

    files
}

fn bench_parse(c: &mut Criterion, dataset: &[(String, String)]) {
    c.bench_function("parse_include", |b| {
        b.iter(|| {
            let mut entries = Entries::default();

            for (name, content) in dataset {
                entries.parse_include(name, content.lines());
            }

            black_box(entries);
        })
    });
}

fn bench_flatten(c: &mut Criterion, dataset: &[(String, String)]) {
    let mut entries = Entries::default();

    for (name, content) in dataset {
        entries.parse_include(name, content.lines());
    }

    c.bench_function("flatten", |b| {
        b.iter(|| {
            let mut out = Vec::new();

            for base in entries.bases() {
                out.push(entries.flatten(&base, None).unwrap());
            }

            black_box(out);
        })
    });
}

fn bench_full(c: &mut Criterion, dataset: &[(String, String)]) {
    c.bench_function("full_pipeline", |b| {
        b.iter(|| {
            let mut entries = Entries::default();

            for (name, content) in dataset {
                entries.parse_include(name, content.lines());
            }

            let mut out = Vec::new();

            for base in entries.bases() {
                out.push(entries.flatten(&base, None).unwrap());
            }

            black_box(out);
        })
    });
}

fn benches(c: &mut Criterion) {
    let dataset = load_dataset(&PathBuf::from(
        env::var("DATASET").expect("DATASET is required"),
    ));

    bench_parse(c, &dataset);
    bench_flatten(c, &dataset);
    bench_full(c, &dataset);
}

criterion_group!(benches_group, benches);
criterion_main!(benches_group);
