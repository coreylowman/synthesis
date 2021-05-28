use chrono::prelude::*;
use serde::Serialize;
use std::fs::File;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub fn train_dir(root: &'static str) -> PathBuf {
    let path = Path::new(root).join(Local::now().format("train_%m-%d-%YT%H-%M-%SZ").to_string());
    if !path.exists() {
        std::fs::create_dir_all(&path).unwrap();
    }
    path
}

pub fn save<T: Serialize>(path: &PathBuf, filename: &'static str, value: &T) {
    save_str(path, filename, &serde_json::to_string(value).expect(""));
}

pub fn save_str(path: &PathBuf, filename: &'static str, value: &String) {
    let mut f = std::fs::File::create(&path.join(filename)).expect("?");
    f.write_all(value.as_bytes()).expect("");
}

pub fn git_hash() -> String {
    let output = Command::new("git")
        .arg("rev-parse")
        .arg("HEAD")
        .output()
        .expect("Unable to get git hash");
    String::from_utf8(output.stdout).unwrap()
}

pub fn git_diff() -> String {
    let output = Command::new("git")
        .arg("diff")
        .output()
        .expect("Unable to get git diff");
    String::from_utf8(output.stdout).unwrap()
}

pub fn add_pgn_result(pgn: &mut File, white_name: &String, black_name: &String, white_reward: f32) {
    write!(pgn, "[White \"{}\"]\n", white_name).expect("");
    write!(pgn, "[Black \"{}\"]\n", black_name).expect("");
    let result = if white_reward == 1.0 {
        // white wins
        "1-0"
    } else if white_reward == -1.0 {
        // black wins
        "0-1"
    } else {
        // draw
        "1/2-1/2"
    };
    write!(pgn, "[Result \"{}\"]\n", result).expect("");
    write!(pgn, "{}\n", result).expect("");
}

pub fn calculate_ratings(dir: &PathBuf) -> Result<(), std::io::Error> {
    let mut child = Command::new("bayeselo.exe")
        .current_dir(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stdin = child.stdin.take().unwrap();
    write!(stdin, "readpgn results.pgn\n")?;
    write!(stdin, "elo\n")?;
    write!(stdin, "mm\n")?;
    write!(stdin, "elo 1 0\n")?;
    write!(stdin, "exactdist\n")?;
    write!(stdin, "ratings >ratings\n")?;
    write!(stdin, "x\n")?;
    write!(stdin, "x\n")?;
    child.wait()?;
    Ok(())
}
