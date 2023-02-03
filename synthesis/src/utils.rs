use chrono::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub fn train_dir(root: &'static str, tag: &'static str) -> std::io::Result<PathBuf> {
    let time = Local::now().format("%m-%d-%YT%H-%M-%SZ").to_string();
    Ok(Path::new(root).join(tag).join(time))
}

pub fn save_str(path: &Path, filename: &'static str, value: &String) -> std::io::Result<()> {
    File::create(path.join(filename)).and_then(|mut f| f.write_all(value.as_bytes()))
}

pub fn git_hash() -> std::io::Result<String> {
    Command::new("git")
        .arg("rev-parse")
        .arg("HEAD")
        .output()
        .map(|output| String::from_utf8(output.stdout).expect("Command didn't produce valid utf-8"))
}

pub fn git_diff() -> std::io::Result<String> {
    Command::new("git")
        .arg("diff")
        .output()
        .map(|output| String::from_utf8(output.stdout).expect("Command didn't produce valid utf-8"))
}

pub fn add_pgn_result(
    pgn: &mut File,
    white_name: &String,
    black_name: &String,
    white_reward: f32,
) -> std::io::Result<()> {
    writeln!(pgn, "[White \"{white_name}\"]")?;
    writeln!(pgn, "[Black \"{black_name}\"]")?;
    let result = if white_reward == 1.0 {
        // white wins
        "1-0"
    } else if white_reward == -1.0 {
        // black wins
        "0-1"
    } else {
        assert_eq!(white_reward, 0.0);
        // draw
        "1/2-1/2"
    };
    writeln!(pgn, "[Result \"{result}\"]")?;
    writeln!(pgn, "{result}")
}

pub fn calculate_ratings(dir: &PathBuf) -> std::io::Result<()> {
    let mut child = Command::new("bayeselo.exe")
        .current_dir(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, "readpgn results.pgn")?;
    writeln!(stdin, "elo")?;
    writeln!(stdin, "mm")?;
    writeln!(stdin, "exactdist")?;
    writeln!(stdin, "ratings >ratings")?;
    writeln!(stdin, "x")?;
    writeln!(stdin, "x")?;
    child.wait()?;
    Ok(())
}

pub fn plot_ratings(dir: &Path) -> std::io::Result<()> {
    let output = Command::new("python")
        .arg("plot_ratings.py")
        .arg(dir.join("ratings").to_str().unwrap())
        .status()?;
    assert!(output.success());
    Ok(())
}

pub fn rankings(dir: &Path) -> std::io::Result<Vec<String>> {
    let file = File::open(dir.join("ratings"))?;
    let reader = std::io::BufReader::new(file);
    let mut names = Vec::new();
    for line in reader.lines().skip(1) {
        let l = String::from(line?.trim());
        if let Some(start_i) = l.find("model_") {
            let end_i = l.find(".ot").unwrap();
            names.push(String::from(l[start_i..end_i + 3].trim()));
        }
    }
    Ok(names)
}
