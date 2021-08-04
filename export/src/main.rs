use base65536;
use std::collections::HashMap;
use std::env;
use std::io::Write;
use std::path::Path;
use tch;

pub fn f32_to_bf16(value: f32) -> u16 {
    // Convert to raw bytes
    let x = value.to_bits();

    // check for NaN
    if x & 0x7FFF_FFFFu32 > 0x7F80_0000u32 {
        // Keep high part of current mantissa but also set most significiant mantissa bit
        return ((x >> 16) | 0x0040u32) as u16;
    }

    // round and shift
    let round_bit = 0x0000_8000u32;
    if (x & round_bit) != 0 && (x & (3 * round_bit - 1)) != 0 {
        (x >> 16) as u16 + 1
    } else {
        (x >> 16) as u16
    }
}

fn serialize_tensor(t: &tch::Tensor) -> String {
    let f32s: Vec<f32> = t.into();
    let u8s: Vec<u8> = f32s
        .iter()
        .map(|f| {
            f32_to_bf16(*f)
                .to_be_bytes()
                .iter()
                .cloned()
                .collect::<Vec<u8>>()
        })
        .flatten()
        .collect();
    base65536::encode(&u8s)
}

fn serialize_tensors<P: AsRef<Path>>(
    variables: &HashMap<String, tch::Tensor>,
    path: P,
) -> Result<(), std::io::Error> {
    let mut names: Vec<String> = variables
        .iter()
        .map(|(k, _)| String::from(k.split(".").next().unwrap()))
        .collect();
    names.sort();
    names.dedup();

    let mut f = std::fs::File::create(path)?;
    let mut i = 0;
    for name in names.iter() {
        let weight_key = format!("{}.weight", name);
        let weight_num_dims = variables.get(&weight_key).unwrap().size().len();
        let bias_key = format!("{}.bias", name);
        f.write_fmt(format_args!(
            "load_{}d(&mut policy.{}, String::from(PARAMETERS[{}]));\n",
            weight_num_dims, weight_key, i,
        ))?;
        i += 1;
        f.write_fmt(format_args!(
            "load_1d(&mut policy.{}, String::from(PARAMETERS[{}]));\n",
            bias_key, i,
        ))?;
        i += 1;
    }

    f.write_fmt(format_args!(
        "const PARAMETERS: [&'static str; {}] = [\n",
        variables.len()
    ))?;
    let mut i = 0;
    for name in names.iter() {
        let weight_key = format!("{}.weight", name);
        let bias_key = format!("{}.bias", name);
        let str_weight = serialize_tensor(&variables.get(&weight_key).unwrap());
        let str_bias = serialize_tensor(&variables.get(&bias_key).unwrap());
        println!("{} - {} {}", name, str_weight.len(), str_bias.len());
        f.write_fmt(format_args!(
            "// {} - {}\n\"{}\",\n\"{}\",\n",
            name, i, str_weight, str_bias,
        ))?;
        i += 2;
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
    let variables: HashMap<String, tch::Tensor> = ts.into_iter().collect();
    serialize_tensors(&variables, dst_params_path)?;
    Ok(())
}
