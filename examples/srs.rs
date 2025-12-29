use std::{fs, path::Path};

use domi::{
    srs::{Rule, RuleSet},
    AttrFilter, Entries,
};

const BASE: &str = "alphabet";

fn main() {
    let data_root = Path::new("data");
    let content = fs::read_to_string(data_root.join(BASE)).unwrap();
    let mut entries = Entries::parse(BASE, content.lines());
    while let Some(i) = entries.next_include() {
        let include = fs::read_to_string(data_root.join(i.as_ref())).unwrap();
        entries.parse_extend(BASE, include.lines());
    }
    // change the `Some(&[AttrFilter::Lacks("attr2")])` to something else can alter behavier,
    // see crate::Entries
    let rule = Rule::from(
        entries
            .flatten(BASE, Some(&[AttrFilter::Lacks("attr2")]))
            .unwrap(),
    );
    // expected output {"version":1,"rules":[{"domain_suffix":["alphabet.com"],"domain_keyword":["fitbit","google"]}]}
    let rule_set = RuleSet::from_iter([rule]);
    println!("{}", serde_json::to_string(&rule_set).unwrap());
}
