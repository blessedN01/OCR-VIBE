path = r"C:\Users\ASUS\Desktop\Projects\OCR-VIBE\src\gui_flet.py"
with open(path, encoding="utf-8") as f:
    s = f.read()

old = '''        "engine_auto": "Auto",
    },
    "fr": {'''
new = '''        "engine_auto": "Auto",
        "settings": "Settings",
        "help": "Help",
        "bounding_box": "Bounding Box Viewer",
        "pdf_thumbnails": "PDF Thumbnails",
        "spell_check": "Spell Check",
        "statistics": "Statistics",
    },
    "fr": {'''

if '"settings":' not in s:
    s2 = s.replace(old, new)
    if s2 != s:
        with open(path, "w", encoding="utf-8") as f:
            f.write(s2)
        print("done")
    else:
        print("not replaced")
else:
    print("exists")
