//! Runs the real `mixel` binary. These tests only exercise paths that exit
//! before the model loads, except `generates_reproducible_image`, which needs
//! the Z-Image-Turbo weights and is ignored by default:
//!
//! ```bash
//! cargo test --release -- --ignored
//! ```

use std::path::Path;
use std::process::{Command, Output};

fn mixel(args: &[&str], cwd: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mixel"))
        .args(args)
        .current_dir(cwd)
        .output()
        .expect("failed to run mixel")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn invalid_jsonl_fails_before_loading_model() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("p.jsonl"),
        "{\"prompt\": \"ok\"}\n{\"prompt\": \"x\", \"width\": 100}\n",
    )
    .unwrap();

    let out = mixel(&["-i", "p.jsonl"], dir.path());
    let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
    assert!(!out.status.success());
    assert!(stderr.contains("1 invalid line(s)"), "{stderr}");
    assert!(stderr.contains("line 2: Image dimensions"), "{stderr}");
    assert!(
        !stdout.contains("Loading"),
        "model should not load:\n{stdout}"
    );
}

#[test]
fn batch_with_all_outputs_present_does_nothing() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("p.jsonl"),
        "{\"prompt\": \"a\", \"seed\": 1}\n{\"prompt\": \"b\", \"output\": \"art/b.png\"}\n",
    )
    .unwrap();
    let out_dir = dir.path().join("out");
    std::fs::create_dir_all(out_dir.join("art")).unwrap();
    std::fs::write(out_dir.join("0001.png"), b"").unwrap();
    std::fs::write(out_dir.join("art/b-42.png"), b"").unwrap(); // random-seed output

    let out = mixel(&["-i", "p.jsonl", "--output-dir", "out"], dir.path());
    let stdout = text(&out.stdout);
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(stdout.contains("Skipping line 1"), "{stdout}");
    assert!(stdout.contains("Skipping line 2"), "{stdout}");
    assert!(stdout.contains("Nothing to do"), "{stdout}");
    assert!(!stdout.contains("Loading"), "{stdout}");
}

#[test]
fn single_mode_rejects_bad_size_before_loading_model() {
    let dir = tempfile::tempdir().unwrap();
    let out = mixel(&["--prompt", "x", "--width", "500"], dir.path());
    assert!(!out.status.success());
    assert!(text(&out.stderr).contains("divisible by 16"));
    assert!(!text(&out.stdout).contains("Loading"));
}

#[test]
fn conflicting_flags_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let out = mixel(&["-i", "p.jsonl", "--prompt", "x"], dir.path());
    assert!(!out.status.success());
    assert!(text(&out.stderr).contains("cannot be used with"));
}

#[test]
fn help_documents_batch_mode() {
    let out = mixel(&["--help"], Path::new("."));
    let stdout = text(&out.stdout);
    assert!(out.status.success());
    for flag in [
        "--input",
        "--output-dir",
        "--overwrite",
        "--seed",
        "USAGE GUIDE",
    ] {
        assert!(stdout.contains(flag), "missing {flag}");
    }
}

#[test]
#[ignore = "needs the ~33 GB Z-Image-Turbo weights and a GPU; run with --ignored"]
fn generates_reproducible_image() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("p.jsonl"),
        concat!(
            "{\"prompt\": \"a red apple\", \"seed\": 5, \"width\": 256, \"height\": 256, \"num_steps\": 2, \"output\": \"a.png\"}\n",
            "{\"prompt\": \"a red apple\", \"seed\": 5, \"width\": 256, \"height\": 256, \"num_steps\": 2, \"output\": \"b.png\"}\n",
            "{\"prompt\": \"a red apple\", \"width\": 320, \"height\": 192, \"num_steps\": 2, \"output\": \"c.png\"}\n",
        ),
    )
    .unwrap();

    let out = mixel(&["-i", "p.jsonl"], dir.path());
    assert!(
        out.status.success(),
        "{}\n{}",
        text(&out.stdout),
        text(&out.stderr)
    );
    assert!(text(&out.stdout).contains("3 generated, 0 skipped, 0 failed"));

    let a = std::fs::read(dir.path().join("a.png")).unwrap();
    let b = std::fs::read(dir.path().join("b.png")).unwrap();
    assert_eq!(a, b, "same seed and prompt should give identical images");
    assert_eq!(
        image::image_dimensions(dir.path().join("a.png")).unwrap(),
        (256, 256)
    );

    // The random-seed image has the seed in its name and the requested size.
    let c = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .find(|n| n.starts_with("c-") && n.ends_with(".png"))
        .expect("random-seed output c-<seed>.png");
    assert!(c[2..c.len() - 4].bytes().all(|b| b.is_ascii_digit()), "{c}");
    assert_eq!(
        image::image_dimensions(dir.path().join(&c)).unwrap(),
        (320, 192)
    );
}

#[test]
#[ignore = "needs the ~33 GB Z-Image-Turbo weights and a GPU; run with --ignored"]
fn img2img_matches_text_to_image_at_full_strength_and_follows_image_size() {
    let dir = tempfile::tempdir().unwrap();
    let base = ["--prompt", "a red apple", "--seed", "5", "--num-steps", "2"];
    // Text-to-image reference at 256x256.
    let out = mixel(
        &[
            &base[..],
            &["--width", "256", "--height", "256", "--output", "t2i.png"],
        ]
        .concat(),
        dir.path(),
    );
    assert!(out.status.success(), "{}", text(&out.stderr));

    // Strength 1.0 ignores the init image's content: byte-identical output.
    let out = mixel(
        &[
            &base[..],
            &[
                "--init-image",
                "t2i.png",
                "--strength",
                "1.0",
                "--output",
                "i2i.png",
            ],
        ]
        .concat(),
        dir.path(),
    );
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert_eq!(
        std::fs::read(dir.path().join("t2i.png")).unwrap(),
        std::fs::read(dir.path().join("i2i.png")).unwrap()
    );

    // Without --width/--height the output follows the init image (320x192).
    image::RgbImage::from_pixel(320, 192, image::Rgb([90, 160, 220]))
        .save(dir.path().join("wide.png"))
        .unwrap();
    let out = mixel(
        &[
            &base[..],
            &[
                "--init-image",
                "wide.png",
                "--strength",
                "0.5",
                "--output",
                "wide-out.png",
            ],
        ]
        .concat(),
        dir.path(),
    );
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(
        text(&out.stdout).contains("strength 0.5, 1 of 2 steps"),
        "{}",
        text(&out.stdout)
    );
    assert_eq!(
        image::image_dimensions(dir.path().join("wide-out.png")).unwrap(),
        (320, 192)
    );
}

#[test]
#[ignore = "needs the ~31 GB Qwen-Image-2.1 weights and a GPU; run with --ignored"]
fn qwen_generates_and_edits() {
    let dir = tempfile::tempdir().unwrap();
    let base = [
        "--model",
        "qwen-image-2.1",
        "--seed",
        "3",
        "--num-steps",
        "2",
    ];
    let out = mixel(
        &[
            &base[..],
            &[
                "--prompt",
                "a red apple",
                "--width",
                "256",
                "--height",
                "192",
                "--output",
                "a.png",
            ],
        ]
        .concat(),
        dir.path(),
    );
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert_eq!(
        image::image_dimensions(dir.path().join("a.png")).unwrap(),
        (256, 192)
    );

    // Editing: the reference image sets the aspect ratio unless a size is given.
    let out = mixel(
        &[
            &base[..],
            &[
                "--prompt",
                "make the apple green",
                "--ref-image",
                "a.png",
                "--width",
                "256",
                "--height",
                "192",
                "--output",
                "b.png",
            ],
        ]
        .concat(),
        dir.path(),
    );
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(text(&out.stdout).contains("Reference image: a.png"));
    assert_eq!(
        image::image_dimensions(dir.path().join("b.png")).unwrap(),
        (256, 192)
    );

    // Z-Image-Turbo refuses reference images before loading anything.
    let out = mixel(&["--prompt", "x", "--ref-image", "a.png"], dir.path());
    assert!(!out.status.success());
    assert!(text(&out.stderr).contains("does not take reference images"));
    assert!(!text(&out.stdout).contains("Loading"));
}
