# domi
domi provides abstractions and utilities for [domain-list-community](https://github.com/v2fly/domain-list-community) data source.

## Example
```rust,no_run
use std::{fs, path::Path};

use domi::Entries;

const BASE: &str = "alphabet";

fn main() {
    let data_root = Path::new("data");

    let content = fs::read_to_string(data_root.join(BASE)).unwrap();

    let mut entries = Entries::parse(BASE, content);

    while let Some(i) = entries.next_include() {
        entries.parse_include_with(i.target(), || {
            fs::read_to_string(data_root.join(i.target())).unwrap()
        });
    }

    println!("{:?}", entries)
}
```

find more examples at [examples/](examples/)

## License

Licensed under either of

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

#### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
<!-- copied from smol's [README.md](https://github.com/smol-rs/smol/tree/1532526ed932495c1b64623043104d567e9fb165?tab=readme-ov-file#license) -->