# domi
domi provides abstractions and utilities for [domain-list-community](https://github.com/v2fly/domain-list-community) data source.

## Example
```rust
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
    // see crate::dlc::Entries
    println!("{:?}", entries.flatten(BASE, Some(&[])).unwrap().dump())
}
```

find more examples at [examples/](examples/)

## License

Licensed under either of

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

#### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
<!-- copied from smol's [README.md](https://github.com/smol-rs/smol/tree/1532526ed932495c1b64623043104d567e9fb165?tab=readme-ov-file#license) -->