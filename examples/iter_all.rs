use std::{fs, path::Path};

use domi::Entries;

fn main() {
    let data_root = Path::new("data");
    let mut entries = Entries::default();
    let read_dir = data_root.read_dir().unwrap();

    for e in read_dir {
        let e = e.unwrap();
        if e.file_type().unwrap().is_file() {
            let name = e.file_name().into_string().unwrap();
            let include = fs::read_to_string(data_root.join(&name)).unwrap();
            entries.parse_include(&name, include.lines());
        }
    }

    let mut out = Vec::new();
    for base in entries.bases() {
        out.push(entries.flatten(&base, None).unwrap());
    }

    println!("{:#?}", out);
}
