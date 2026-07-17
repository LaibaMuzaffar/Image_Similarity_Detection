import os
import glob
import numpy as np
import torch
import clip
import imagehash
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from insightface.app import FaceAnalysis

# Terminal Styling Constants
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# =====================================================================
# 1. INITIALIZATION & MODEL LOADING
# =====================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{BLUE}🔄 Initializing AI Engine on device: {device.upper()}{RESET}...")

face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)


# =====================================================================
# 2. AI EVALUATION PIPELINE FUNCTIONS
# =====================================================================
def get_cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_face_embedding(image_path):
    try:
        img = np.array(Image.open(image_path).convert("RGB"))
        faces = face_app.get(img)
        if not faces: 
            return None
        faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
        return faces[0].embedding
    except Exception as e:
        print(f"{YELLOW}⚠️  Biometric extraction bypassed on {os.path.basename(image_path)}: {e}{RESET}")
        return None

def get_clip_embedding(image_path):
    image = clip_preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(image)
        emb /= emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()


# =====================================================================
# 3. GUI POPUP DISPLAY INTERFACE (UPGRADED STYLISH EDITION)
# =====================================================================
def show_gui_result(img1_path, img2_path, score, category, logs_dict):
    """Creates a premium, polished desktop interface to show target comparison metrics."""
    root = tk.Tk()
    root.title("Vision Similarity Hub")
    root.geometry("820x620")
    root.configure(bg="#0F0F12")  # Deep cyber-slate background
    root.resizable(False, False)

    # Global Style Setup
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(".", background="#0F0F12", foreground="#FFFFFF")

    # Dynamic Theme Color Assigner based on confidence
    if score >= 80:
        accent_color, status_text = "#00FF66", "HIGH SIMILARITY MATCH"
    elif score >= 50:
        accent_color, status_text = "#FFCC00", "PARTIAL ALIGNMENT DETECTED"
    else:
        accent_color, status_text = "#FF3366", "DISTINCT / UNRELATED CONTENT"

    # Top Header Banner
    header_frame = tk.Frame(root, bg="#16161D", height=60)
    header_frame.pack(fill="x", side="top")
    header_frame.pack_propagate(False)
    
    # Removed letterspace option and manually spaced out the characters slightly for the premium look
    header_title = tk.Label(
        header_frame, text="C O M P U T E R   V I S I O N   M A T R I X   R E S U L T S",
        font=("Segoe UI", 11, "bold"), fg="#8F90A6", bg="#16161D"
    )
    header_title.pack(side="left", padx=25, pady=18)

    pipeline_badge = tk.Label(
        header_frame, text=f"MODE: {category.upper()}",
        font=("Consolas", 9, "bold"), fg="#FFFFFF", bg="#282836", padx=10, pady=4
    )
    pipeline_badge.pack(side="right", padx=25, pady=16)

    # Main Grid Layout Container
    main_container = tk.Frame(root, bg="#0F0F12")
    main_container.pack(fill="both", expand=True, padx=25, pady=20)
    
    main_container.columnconfigure(0, weight=4)
    main_container.columnconfigure(1, weight=3)

    # --- LEFT SIDE: COMPARISON IMAGES VIEWPORT ---
    viewports_frame = tk.Frame(main_container, bg="#0F0F12")
    viewports_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

    def create_image_card(parent, path, label_text):
        card = tk.Frame(parent, bg="#16161D", highlightbackground="#232330", highlightthickness=1)
        card.pack(fill="x", pady=(0, 15))
        
        # Load and cleanly scale images keeping proportions
        raw_img = Image.open(path)
        raw_img.thumbnail((220, 160))
        
        thumb = ImageTk.PhotoImage(raw_img)
        
        img_label = tk.Label(card, image=thumb, bg="#16161D")
        img_label.image = thumb  # Keep hard reference link
        img_label.pack(side="left", padx=15, pady=15)

        info_frame = tk.Frame(card, bg="#16161D")
        info_frame.pack(side="left", fill="both", expand=True, padx=5, pady=15)

        tk.Label(info_frame, text=label_text, font=("Segoe UI", 9, "bold"), fg=accent_color, bg="#16161D").pack(anchor="w")
        tk.Label(info_frame, text=os.path.basename(path), font=("Segoe UI", 10, "bold"), fg="#FFFFFF", bg="#16161D", wraplength=180, justify="left").pack(anchor="w", pady=(2, 5))
        
        size_txt = f"Resolution: {raw_img.width}x{raw_img.height}"
        tk.Label(info_frame, text=size_txt, font=("Consolas", 8), fg="#636375", bg="#16161D").pack(anchor="w")

    create_image_card(viewports_frame, img1_path, "TARGET OBJECT ALPHA")
    create_image_card(viewports_frame, img2_path, "TARGET OBJECT BETA")

    # --- RIGHT SIDE: ANALYTICS HUB & CIRCULAR DIAL ---
    analytics_frame = tk.Frame(main_container, bg="#16161D", highlightbackground="#232330", highlightthickness=1)
    analytics_frame.grid(row=0, column=1, sticky="nsew")

    # Visual Circular Meter Progress via canvas drawing layer
    canvas = tk.Canvas(analytics_frame, width=160, height=160, bg="#16161D", highlightthickness=0)
    canvas.pack(pady=(25, 10))
    
    # Draw background track circle
    canvas.create_arc(15, 15, 145, 145, start=0, extent=359, outline="#232330", width=10, style="arc")
    # Draw interactive active accent arc layer based on evaluation result
    extent_angle = -(score / 100.0) * 359
    canvas.create_arc(15, 15, 145, 145, start=90, extent=extent_angle, outline=accent_color, width=10, style="arc")
    
    # Numerical inner text label
    canvas.create_text(80, 80, text=f"{int(score)}%", fill="#FFFFFF", font=("Segoe UI", 26, "bold"))

    # Status Message Container Banner
    tk.Label(analytics_frame, text=status_text, font=("Segoe UI", 9, "bold"), fg=accent_color, bg="#16161D").pack()

    # Diagnostic Data Breakdown Matrix Separator
    divider = tk.Frame(analytics_frame, bg="#232330", height=1)
    divider.pack(fill="x", padx=20, pady=20)

    tk.Label(analytics_frame, text="DIAGNOSTIC MATRIX LOGS", font=("Segoe UI", 8, "bold"), fg="#636375", bg="#16161D").pack(anchor="w", padx=20)

    # Programmatically inject rows into logging stack box UI
    for key, val in logs_dict.items():
        row = tk.Frame(analytics_frame, bg="#16161D")
        row.pack(fill="x", padx=20, pady=6)
        tk.Label(row, text=key, font=("Segoe UI", 9), fg="#8F90A6", bg="#16161D").pack(side="left")
        tk.Label(row, text=val, font=("Consolas", 9, "bold"), fg="#FFFFFF", bg="#16161D").pack(side="right")

    # Bottom status system confirmation message
    footer = tk.Label(
        root, text="System Core status: Verified operational alignment pipeline.", 
        font=("Segoe UI", 8), fg="#424252", bg="#0F0F12", pady=8
    )
    footer.pack(side="bottom")

    root.mainloop()


# =====================================================================
# 4. MAIN RUNTIME ENGINE
# =====================================================================
def run_standalone_comparison():
    print("\n" + "═"*60)
    print(f" {BOLD}{CYAN}🤖 VISION PIPELINE: DESKTOP SCAN & COMPARE{RESET} ")
    print("═"*60)
    
    user_profile = os.environ.get("USERPROFILE", "C:\\Users\\Zohaib")
    standard_desktop = os.path.join(user_profile, "Desktop")
    onedrive_desktop = os.path.join(user_profile, "OneDrive", "Desktop")
    
    desktop_path = onedrive_desktop if os.path.exists(onedrive_desktop) else standard_desktop
    print(f" 📂 {BOLD}Target Path:{RESET} {desktop_path}")
    
    img1_matches = glob.glob(os.path.join(desktop_path, "download (4).*")) + glob.glob(os.path.join(desktop_path, "download(4).*"))
    img2_matches = glob.glob(os.path.join(desktop_path, "download (5).*")) + glob.glob(os.path.join(desktop_path, "download(5).*"))
    
    if not img1_matches or not img2_matches:
        print("\n" + "─"*60)
        print(f" {RED}❌ CRITICAL ERROR: Asset Detection Failed{RESET} ")
        print("─"*60)
        return

    path1 = img1_matches[0]
    path2 = img2_matches[0]
    
    print(f" • {GREEN}Found Asset 1:{RESET} {os.path.basename(path1)}")
    print(f" • {GREEN}Found Asset 2:{RESET} {os.path.basename(path2)}")
    print("─"*60)

    try:
        logs_dict = {}
        # --- PIPELINE SEGMENT 1: BIOMETRIC RECOGNITION (FACE FOCUS) ---
        f1, f2 = get_face_embedding(path1), get_face_embedding(path2)
        if f1 is not None and f2 is not None:
            category = "Biometric (Face)"
            raw_sim = get_cosine_similarity(f1, f2)
            
            if raw_sim > 0.60: final_sim = np.interp(raw_sim, [0.60, 1.0], [92, 100])
            elif raw_sim > 0.38: final_sim = np.interp(raw_sim, [0.38, 0.60], [65, 92])
            else: final_sim = np.interp(raw_sim, [0.0, 0.38], [0, 65])
            
            score = round(final_sim, 2)
            logs_dict["Raw Face Cosine"] = f"{raw_sim:.4f}"
            
        else:
            # --- PIPELINE SEGMENT 2: RECALIBRATED GENERAL SEMANTIC ALIGNMENT (CLIP FOCUS) ---
            category = "General Context/Object"
            c_sim = get_cosine_similarity(get_clip_embedding(path1), get_clip_embedding(path2))
            h1, h2 = imagehash.phash(Image.open(path1)), imagehash.phash(Image.open(path2))
            hash_sim = 1 - (h1 - h2) / 64.0

            if c_sim > 0.85:
                semantic_score = np.interp(c_sim, [0.85, 1.0], [85, 100])
            elif c_sim > 0.70:
                semantic_score = np.interp(c_sim, [0.70, 0.85], [40, 85])
            elif c_sim > 0.60:
                semantic_score = np.interp(c_sim, [0.60, 0.70], [10, 40])
            else:
                semantic_score = np.interp(c_sim, [0.40, 0.60], [0, 10])

            final_sim = (semantic_score * 0.9) + (hash_sim * 10.0)
            score = round(min(max(final_sim, 0.0), 100.0), 2)
            
            logs_dict["Deep CLIP Vector"] = f"{c_sim:.4f}"
            logs_dict["Perceptual Hash"] = f"{hash_sim:.4f}"

        if score >= 80: color_tag = GREEN
        elif score >= 50: color_tag = YELLOW
        else: color_tag = RED

        print(f"\n💸 Console Calculation Complete. Launching Modern Dark Matrix Panel Window...")
        print(f"🎉 FINAL SIMILARITY MATCH: {color_tag}{score}%{RESET}\n")
        
        # Trigger Premium Form
        show_gui_result(path1, path2, score, category, logs_dict)

    except Exception as e:
        import traceback
        print(f"\n{RED}❌ Operational Pipeline Interrupted:{RESET}")
        traceback.print_exc()

if __name__ == "__main__":
    run_standalone_comparison()