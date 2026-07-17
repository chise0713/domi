use std::io::Result;

fn main() -> Result<()> {
    if cfg!(feature = "rustc-hash") && cfg!(feature = "ahash") {
        println!("cargo:warning=both ahash and rustc-hash enabled; using rustc-hash");
    }
    #[cfg(feature = "prost")]
    {
        const GEOSITE_PROTO: &str = "proto/geosite.proto";
        println!("cargo:rerun-if-changed={}", GEOSITE_PROTO);
        prost_build::compile_protos(&[GEOSITE_PROTO], &["proto/"])?;
    }
    Ok(())
}
