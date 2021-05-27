use chrono::prelude::*;
use serde::Serialize;
use std::fs::File;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::process::Command;

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
