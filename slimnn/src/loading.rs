use base65536;

fn bytes_to_floats(bytes: Vec<u8>) -> Vec<f32> {
    let mut floats = Vec::with_capacity(bytes.len() / 4);
    assert!(bytes.len() % 4 == 0);
    for i in (0..bytes.len()).step_by(4) {
        floats.push(f32::from_be_bytes([
            bytes[i],
            bytes[i + 1],
            bytes[i + 2],
            bytes[i + 3],
        ]));
    }
    floats
}

pub fn load_1d<const N: usize>(data: &mut [f32; N], params: String) {
    let bytes = base65536::decode(params);
    let floats = bytes_to_floats(bytes);
    assert_eq!(floats.len(), N);
    unsafe { std::ptr::copy(floats.as_ptr(), data.as_mut_ptr(), floats.len()) };
}

pub fn load_2d<const I: usize, const O: usize>(data: &mut [[f32; I]; O], params: String) {
    let bytes = base65536::decode(params);
    let floats = bytes_to_floats(bytes);
    assert_eq!(floats.len(), I * O);
    unsafe { std::ptr::copy(floats.as_ptr(), data[0].as_mut_ptr(), floats.len()) };
}

pub fn load_4d<const I: usize, const J: usize, const K: usize, const L: usize>(
    data: &mut [[[[f32; I]; J]; K]; L],
    params: String,
) {
    let bytes = base65536::decode(params);
    let floats = bytes_to_floats(bytes);
    assert_eq!(floats.len(), I * J * K * L);
    unsafe { std::ptr::copy(floats.as_ptr(), data[0][0][0].as_mut_ptr(), floats.len()) };
}
