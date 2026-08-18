// The Rust shell hosts the native window AND launches/manages the Python
// backend as a child process, so the end-user experience is "run the app,
// it just works" — no separate terminal/python command needed after the
// one-time environment setup (setup.bat) has been run somewhere on disk.
//
// The installed .exe and the backend/ folder (with its multi-GB venv) live
// in different locations — Tauri's installer doesn't bundle the ML
// dependencies (see README). So on first launch we ask the user, once, to
// locate their backend/ folder via a native picker, remember that choice,
// and spawn python -m server.ws_server from it on every future launch.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::{Manager, State};
use tauri_plugin_dialog::DialogExt;

#[derive(Serialize, Deserialize, Default)]
struct AppConfig {
    backend_path: Option<String>,
}

struct BackendProcess(Mutex<Option<Child>>);

fn config_file_path(app: &tauri::AppHandle) -> PathBuf {
    let dir = app
        .path()
        .app_config_dir()
        .expect("no app config dir available");
    let _ = fs::create_dir_all(&dir);
    dir.join("config.json")
}

fn load_config(app: &tauri::AppHandle) -> AppConfig {
    let path = config_file_path(app);
    fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_config(app: &tauri::AppHandle, config: &AppConfig) {
    let path = config_file_path(app);
    if let Ok(json) = serde_json::to_string_pretty(config) {
        let _ = fs::write(path, json);
    }
}

fn python_exe_in(backend_path: &Path) -> PathBuf {
    if cfg!(windows) {
        backend_path.join("venv").join("Scripts").join("python.exe")
    } else {
        backend_path.join("venv").join("bin").join("python")
    }
}

/// A "valid" backend folder: setup.bat/.sh has actually been run there
/// (venv exists with a python executable) and it's really this project's
/// backend (server/ws_server.py present), not some unrelated folder.
fn is_valid_backend_path(p: &Path) -> bool {
    python_exe_in(p).is_file() && p.join("server").join("ws_server.py").is_file()
}

fn spawn_backend(backend_path: &Path) -> Option<Child> {
    let python = python_exe_in(backend_path);
    let mut cmd = Command::new(python);
    cmd.arg("-u").arg("-m").arg("server.ws_server");
    cmd.current_dir(backend_path);

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // Suppress the console window that spawning a Python process would
        // otherwise pop up alongside the app window.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    match cmd.spawn() {
        Ok(child) => Some(child),
        Err(e) => {
            eprintln!("Vertumnus: failed to start backend process: {e}");
            None
        }
    }
}

/// Blocks the setup hook (fine — this runs before the window is shown)
/// asking the user to locate backend/ until they pick a folder that
/// actually looks set up, or cancel (in which case the app still opens,
/// just showing "backend disconnected" until they fix it and relaunch).
fn resolve_backend_path(app: &tauri::AppHandle, config: &mut AppConfig) -> Option<PathBuf> {
    if let Some(p) = &config.backend_path {
        let path = PathBuf::from(p);
        if is_valid_backend_path(&path) {
            return Some(path);
        }
    }

    loop {
        let picked = app
            .dialog()
            .file()
            .set_title("Select your Vertumnus 'backend' folder (after running setup.bat there)")
            .blocking_pick_folder();

        let folder = match picked {
            Some(f) => f,
            None => return None, // user cancelled
        };

        let path = match folder.into_path() {
            Ok(p) => p,
            Err(_) => continue,
        };

        if is_valid_backend_path(&path) {
            config.backend_path = Some(path.to_string_lossy().to_string());
            save_config(app, config);
            return Some(path);
        }

        eprintln!(
            "Vertumnus: '{}' doesn't look like a set-up backend folder \
             (expected venv + server/ws_server.py there). Run setup.bat \
             first, then pick the folder again.",
            path.display()
        );
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(BackendProcess(Mutex::new(None)))
        .setup(|app| {
            let handle = app.handle().clone();
            let mut config = load_config(&handle);

            if let Some(backend_path) = resolve_backend_path(&handle, &mut config) {
                if let Some(child) = spawn_backend(&backend_path) {
                    let state: State<BackendProcess> = handle.state();
                    *state.0.lock().unwrap() = Some(child);
                }
            }

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let state: State<BackendProcess> = window.state();
                let mut guard = state.0.lock().unwrap();
                if let Some(mut child) = guard.take() {
                    let _ = child.kill();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Vertumnus");
}
