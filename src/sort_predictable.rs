use super::*;

const VARIANT_LEN: usize = 6;
const CONTENS: [&str; VARIANT_LEN] = [
    "full:full\nkeyword:keyword\nregexp:regexp",
    "full:full\nregexp:regexp\nkeyword:keyword",
    "keyword:keyword\nfull:full\nregexp:regexp",
    "keyword:keyword\nregexp:regexp\nfull:full",
    "regexp:regexp\nfull:full\nkeyword:keyword",
    "regexp:regexp\nkeyword:keyword\nfull:full",
];

pub(crate) fn helper<T, K, F, C>(mut build: F, mut cmp: C)
where
    K: Ord,
    F: FnMut(FlatDomains) -> T,
    C: FnMut(&T) -> K,
{
    let list: [T; VARIANT_LEN] = std::array::from_fn(|i| {
        let domains = Entries::parse(BASE, CONTENS[i].lines())
            .flatten(BASE, None)
            .unwrap();

        build(domains)
    });

    assert!(list.windows(2).all(|w| cmp(&w[0]) == cmp(&w[1])));
}
