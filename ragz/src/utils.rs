use chrono::prelude::*;
use serde::Serialize;
use std::fs::File;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub fn train_dir(root: &'static str, tag: &'static str) -> std::io::Result<PathBuf> {
    let time = Local::now().format("%m-%d-%YT%H-%M-%SZ").to_string();
    let path = Path::new(root).join(tag).join(time);
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

pub fn save<T: Serialize>(
    path: &PathBuf,
    filename: &'static str,
    value: &T,
) -> std::io::Result<()> {
    save_str(path, filename, &serde_json::to_string(value)?)
}

pub fn save_str(path: &PathBuf, filename: &'static str, value: &String) -> std::io::Result<()> {
    std::fs::File::create(&path.join(filename)).and_then(|mut f| f.write_all(value.as_bytes()))
}

pub fn git_hash() -> std::io::Result<String> {
    Command::new("git")
        .arg("rev-parse")
        .arg("HEAD")
        .output()
        .and_then(|output| {
            Ok(String::from_utf8(output.stdout).expect("Command didn't produce valid utf-8"))
        })
}

pub fn git_diff() -> std::io::Result<String> {
    Command::new("git").arg("diff").output().and_then(|output| {
        Ok(String::from_utf8(output.stdout).expect("Command didn't produce valid utf-8"))
    })
}

pub fn add_pgn_result(
    pgn: &mut File,
    white_name: &String,
    black_name: &String,
    white_reward: f32,
) -> std::io::Result<()> {
    write!(pgn, "[White \"{}\"]\n", white_name)?;
    write!(pgn, "[Black \"{}\"]\n", black_name)?;
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
    write!(pgn, "[Result \"{}\"]\n", result)?;
    write!(pgn, "{}\n", result)
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
    write!(stdin, "exactdist\n")?;
    write!(stdin, "ratings >ratings\n")?;
    write!(stdin, "x\n")?;
    write!(stdin, "x\n")?;
    child.wait()?;
    Ok(())
}
