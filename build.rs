use std::io::Result;

fn main() -> Result<()> {
    #[cfg(feature = "protobuf")]
    {
        const GEOSITE_PROTO: &str = "proto/geosite.proto";
        println!("cargo:rerun-if-changed={}", GEOSITE_PROTO);
        prost_build::compile_protos(&[GEOSITE_PROTO], &["proto/"])?;
    }
    Ok(())
}
