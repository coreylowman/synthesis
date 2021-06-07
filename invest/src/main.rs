use base65536;
use std::env;
use std::io::Write;
use std::path::Path;
use tch;

fn serialize_tensor(t: &tch::Tensor) -> String {
    let f32s: Vec<f32> = t.into();
    let u8s: Vec<u8> = f32s
        .iter()
        .map(|f| f.to_be_bytes().iter().cloned().collect::<Vec<u8>>())
        .flatten()
        .collect();
    base65536::encode(&u8s)
}

fn serialize_tensors<P: AsRef<Path>>(
    vs: &Vec<(String, tch::Tensor)>,
    path: P,
) -> Result<(), std::io::Error> {
    let mut f = std::fs::File::create(path)?;
    f.write_fmt(format_args!(
        "const PARAMETERS: [&'static str; {}] = [\n",
        vs.len()
    ))?;
    for (i, (key, value)) in vs.iter().enumerate() {
        let str_tensor = serialize_tensor(&value);
        println!("{} - {}", key, str_tensor.len());
        f.write_fmt(format_args!("// {} - {}\n\"{}\",\n", key, i, str_tensor,))?;
    }
    f.write(b"];\n")?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    assert!(args.len() == 3);

    let src_varstore_path = &args[1];
    let dst_params_path = &args[2];

    let ts = tch::Tensor::load_multi(src_varstore_path)?;
    serialize_tensors(&ts, dst_params_path)?;
    Ok(())
}
