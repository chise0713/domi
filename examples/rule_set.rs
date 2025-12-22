use std::{fs, path::Path};

use domi::{
    srs::{Rule, RuleSet},
    AttrFilter, Entries,
};

const BASE: &str = "alphabet";

fn main() {
    let data_root = Path::new("data");
    let content = fs::read_to_string(data_root.join(BASE)).unwrap();
    let mut entries = Entries::parse(BASE, content.lines()).unwrap();
    while let Some(i) = entries.next_include() {
        let include = fs::read_to_string(data_root.join(i.as_ref())).unwrap();
        entries.parse_extend(include.lines()).unwrap();
    }
    // change the `Some(&[AttrFilter::Lacks("attr2")])` to something else can alter behavier,
    // see crate::dlc::Entries
    let rule = Rule::from(
        entries
            .flatten(BASE, Some(&[AttrFilter::Lacks("attr2")]))
            .unwrap(),
    );
    // expected domain_suffix: Some(["alphabet.com"]), domain_keyword: Some(["fitbit", "google"])
    println!("{:?}", RuleSet::from_iter([rule]));
}
