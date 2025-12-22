use std::{fs, path::Path};

use domi::Entries;

const BASE: &str = "alphabet";

fn main() {
    let data_root = Path::new("data");
    let content = fs::read_to_string(data_root.join(BASE)).unwrap();
    let mut entries = Entries::parse(BASE, content.lines()).unwrap();
    while let Some(i) = entries.next_include() {
        let include = fs::read_to_string(data_root.join(i.as_ref())).unwrap();
        entries.parse_extend(include.lines()).unwrap();
    }
    // expect: domain_keyword: ["fitbit", "google"]
    // change the `Some(&[])` to something else can alter behavier,
    // see crate::Entries
    println!("{:?}", entries.flatten(BASE, Some(&[])).unwrap().dump());
}
