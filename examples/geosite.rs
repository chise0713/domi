use std::{fs, path::Path};

use domi::{
    geosite::proto::{GeoSite, GeoSiteList},
    AttrFilter, Entries,
};
use prost::Message as _;

const BASE: &str = "alphabet";

fn main() {
    let data_root = Path::new("data");
    let content = fs::read_to_string(data_root.join(BASE)).unwrap();
    let mut entries = Entries::parse(BASE, content.lines());
    while let Some(i) = entries.next_include() {
        let include = fs::read_to_string(data_root.join(i.as_ref())).unwrap();
        entries.parse_extend(include.lines());
    }
    // change the `Some(&[AttrFilter::Lacks("attr2")])` to something else can alter behavier,
    // see crate::Entries
    let flattened = entries
        .flatten(BASE, Some(&[AttrFilter::Lacks("attr2")]))
        .unwrap();
    let geosite = GeoSiteList::from_iter([GeoSite::from(flattened)]);
    let out = geosite.encode_to_vec();
    println!("{:?}", String::from_utf8_lossy(&out));
}
