"""Georgian user-interface strings. Code and identifiers stay in English."""

from __future__ import annotations

T: dict[str, str] = {
    # application
    "app_title": "MathVision 3D — მათემატიკური ვიზუალიზაციის ლაბორატორია",
    "file": "ფაილი", "view": "ხედი", "help": "დახმარება",
    "load_image": "სურათის ჩატვირთვა…", "load_sample": "ნიმუშის ჩატვირთვა",
    "load_video": "ვიდეოფაილის გახსნა (კამერის ნაცვლად)…",
    "save_screenshot": "3D ხედის სურათის შენახვა…", "export_object": "ობიექტის ექსპორტი…",
    "exit": "გასვლა", "about": "პროგრამის შესახებ", "usage": "გამოყენების ინსტრუქცია",

    # camera panel
    "camera": "კამერა", "camera_select": "კამერის არჩევა", "camera_start": "ჩართვა",
    "camera_stop": "გამორთვა", "camera_none": "კამერა ვერ მოიძებნა",
    "camera_searching": "კამერების ძიება…", "camera_index": "კამერა №{}",
    "freeze": "კადრის დაფიქსირება", "unfreeze": "ცოცხალი რეჟიმი",
    "capture_extract": "გადაღება და ამოღება", "extract_selected": "არჩეული ობიექტის ამოღება",
    "manual_region": "რეგიონის ხელით არჩევა", "clear_region": "რეგიონის გასუფთავება",
    "detected_objects": "აღმოჩენილი ობიექტები", "preview_mode": "ხედის რეჟიმი",
    "preview_camera": "კამერა", "preview_rectified": "გასწორებული ფურცელი",
    "preview_binary": "ბინარული გამოსახულება", "show_overlays": "აღმოჩენების ჩვენება",
    "recognition_settings": "ამოცნობის პარამეტრები", "threshold_mode": "ზღვრის რეჟიმი",
    "thr_adaptive": "ადაპტური", "thr_otsu": "ოცუ (ავტომატური)", "thr_manual": "ხელით",
    "threshold": "ზღვარი", "sensitivity": "მელნის მგრძნობელობა", "block_size": "ლოკალური ფანჯარა",
    "contrast": "კონტრასტი", "noise_reduction": "ხმაურის შემცირება",
    "edge_sensitivity": "კიდეების მგრძნობელობა", "min_size": "ობიექტის მინ. ზომა",
    "remove_shadows": "ჩრდილების მოცილება", "detect_paper": "ფურცლის ამოცნობა და პერსპექტივის კორექცია",
    "detect_axes": "კოორდინატთა ღერძების ამოცნობა", "ignore_border": "კიდეზე მდებარე ობიექტების უგულებელყოფა",
    "reset_defaults": "ნაგულისხმევი პარამეტრები",
    "gestures": "ჟესტებით მართვა (ექსპერიმენტული)", "gestures_enable": "ჟესტების ჩართვა",
    "gesture_mode": "მოძრაობის რეჟიმი", "gesture_rotate": "ბრუნვა", "gesture_translate": "გადაადგილება",
    "gesture_unavailable": "ჟესტები მიუწვდომელია: {}",

    # viewport
    "reset_camera": "კამერის აღდგენა", "grid": "ბადე", "axes": "ღერძები",
    "perspective": "პერსპექტივა", "orthographic": "ორთოგრაფიული",
    "mode_camera": "კამერის მართვა", "mode_object": "ობიექტის მართვა",
    "view_front": "წინიდან (XY)", "view_top": "ზემოდან (XZ)", "view_side": "გვერდიდან (YZ)",
    "view_iso": "იზომეტრია", "auto_rotate": "ავტო-ბრუნვა",

    # controls panel
    "workspace_objects": "სამუშაო სივრცის ობიექტები", "delete": "წაშლა", "clear_all": "ყველას წაშლა",
    "object_info": "ობიექტის ინფორმაცია", "type": "ტიპი", "equation": "განტოლება",
    "confidence": "სანდოობა", "fit_error": "მიახლოების ცდომილება (NRMSE)",
    "candidates": "კანდიდატი მოდელები", "coordinates_status": "კოორდინატები",
    "calibrated": "დაკალიბრებული", "normalized": "ნორმალიზებული — შეფასებითი ფორმა",
    "analysis": "მათემატიკური ანალიზი", "display": "ჩვენების რეჟიმი",
    "mode_flat": "ბრტყელი ობიექტი 3D სივრცეში", "mode_extrude": "ექსტრუზია (პრიზმა / ცილინდრი)",
    "mode_analysis": "გეომეტრიული ანალიზი", "mode_curve2d": "2D მრუდი XY სიბრტყეში",
    "mode_space": "სივრცითი მრუდი (x, f(x), z)", "mode_surface": "მათემატიკური ზედაპირი",
    "extrude_depth": "სიღრმე", "space_z": "z(x, y) =", "surface_type": "ზედაპირის ტიპი",
    "surf_translational": "z = f(x), y ∈ [a, b]", "surf_product": "z = f(x)·f(y)",
    "surf_radial": "z = f(r), r = √(x²+y²)", "surf_revolution": "ბრუნვის ზედაპირი X ღერძის გარშემო",
    "y_range": "y დიაპაზონი", "show_derivative": "წარმოებულის ჩვენება f′(x)",
    "show_tangent": "მხები წრფე", "tangent_at": "მხების წერტილი x₀",
    "show_samples": "ამოღებული წერტილების ჩვენება",
    "transform_notice": "⚠ ექსტრუზია, სივრცითი მრუდი და ზედაპირი მომხმარებლის მიერ არჩეული "
                        "მათემატიკური გარდაქმნაა. ერთი 2D ესკიზი ცალსახად არ განსაზღვრავს რეალურ 3D ობიექტს.",
    "transform": "გარდაქმნა", "rotation": "ბრუნვა (°)", "position": "მდებარეობა", "scale": "მასშტაბი",
    "reset_orientation": "ორიენტაციის აღდგენა", "restore_original": "საწყისის აღდგენა",
    "rendering": "რენდერინგი", "style": "სტილი", "style_surface": "მყარი",
    "style_wireframe": "კარკასი", "style_points": "წერტილები", "style_edges": "მყარი + წიბოები",
    "color": "ფერი", "line_width": "ხაზის სისქე", "opacity": "გაუმჭვირვალობა",
    "show_vertices": "წვეროების ჩვენება", "show_labels": "წარწერების ჩვენება",
    "calibration": "კალიბრაცია", "reference_length": "საცნობარო სიგრძე", "unit": "ერთეული",
    "x_start": "x-ის საწყისი მნიშვნელობა", "apply": "გამოყენება", "clear_calibration": "კალიბრაციის მოხსნა",
    "reference_is": "საცნობარო ზომა: {}",
    "ref_side_ab": "გვერდი AB", "ref_diameter": "დიამეტრი", "ref_major_axis": "დიდი ღერძი",
    "ref_length": "მონაკვეთის სიგრძე", "ref_chord": "ქორდა", "ref_x_span": "მრუდის სიგანე x-ღერძზე",
    "no_object": "ობიექტი არ არის არჩეული",

    # status / messages
    "status": "სტატუსი", "detection": "ამოცნობა", "fps": "კადრი/წმ", "messages": "შეტყობინებები",
    "ready": "მზადაა", "camera_opened": "კამერა ჩაირთო: №{}", "camera_failed": "კამერა №{} ვერ გაიხსნა",
    "camera_stopped": "კამერა გამორთულია", "image_loaded": "სურათი ჩაიტვირთა: {}",
    "image_failed": "სურათის წაკითხვა ვერ მოხერხდა: {}",
    "objects_found": "აღმოჩენილია {} ობიექტი", "no_objects": "ობიექტები ვერ მოიძებნა",
    "paper_found": "ფურცელი ნაპოვნია", "axes_found": "ღერძები ნაპოვნია",
    "extracted": "ამოღებულია: {}", "select_first": "ჯერ აირჩიეთ ობიექტი (დააწკაპუნეთ ვიდეოზე ან სიაში)",
    "too_dark": "სურათი ძალიან მუქია — გააუმჯობესეთ განათება",
    "frozen": "კადრი დაფიქსირებულია", "live": "ცოცხალი რეჟიმი",
    "roi_hint": "მონიშნეთ მართკუთხა რეგიონი ვიდეოზე მაუსით",
    "roi_set": "რეგიონი არჩეულია", "calibration_applied": "კალიბრაცია გამოყენებულია",
    "expression_error": "გამოსახულების შეცდომა: {}", "saved": "შენახულია: {}",
    "gestures_on": "ჟესტები ჩართულია", "gestures_off": "ჟესტები გამორთულია",
    "gesture_state": "ჟესტი: {}", "no_video": "ვიდეო არ არის — ჩართეთ კამერა ან ჩატვირთეთ სურათი",

    # analysis labels
    "vertices": "წვეროები", "sides": "გვერდები", "angles": "კუთხეები", "perimeter": "პერიმეტრი",
    "area": "ფართობი", "angle_sum": "კუთხეების ჯამი", "triangle_type": "სამკუთხედის ტიპი",
    "center": "ცენტრი", "radius": "რადიუსი", "semi_axes": "ნახევარღერძები", "eccentricity": "ექსცენტრისიტეტი",
    "circumference": "წრეწირის სიგრძე", "orientation": "ორიენტაცია", "length": "სიგრძე",
    "slope": "დახრა", "endpoints": "ბოლოები", "central_angle": "ცენტრალური კუთხე",
    "arc_length": "რკალის სიგრძე", "domain": "განსაზღვრის არე (შეფასება)",
    "range": "მნიშვნელობათა არე (შეფასება)", "curve_length": "მრუდის სიგრძე (რიცხვითი)",
    "samples": "ამოღებული წერტილები", "parameters": "პარამეტრები", "period": "პერიოდი",
    "tangent_eq": "მხები", "bbox": "საზღვრები", "derivative_na": "წარმოებული განსაზღვრული არ არის ამ მრუდისთვის",
    "units_normalized": "ნ.ე.", "pixels": "პიქს.", "measure_note_norm":
        "ზომები ნორმალიზებულ ერთეულებშია (შეფასება). სიზუსტისთვის გამოიყენეთ კალიბრაცია.",
    "measure_note_cal": "ზომები დაკალიბრებულ ერთეულებშია ({}).",
    "equilateral": "ტოლგვერდა", "isosceles": "ტოლფერდა", "scalene": "სხვადასხვაგვერდა",
    "acute": "მახვილკუთხა", "right": "მართკუთხა", "obtuse": "ბლაგვკუთხა",
    "estimated_shape": "შეფასებითი ფორმა (ნორმალიზებული კოორდინატები)",
    "freehand_note": "მრუდი ცნობილ ფუნქციას საიმედოდ არ ემთხვევა — შენახულია როგორც ინტერპოლირებული მრუდი.",
    "phase_note": "ფაზა გამოთვლილია შეფასებითი კოორდინატთა სათავის მიმართ.",
}

KIND_NAMES = {
    "triangle": "სამკუთხედი", "rectangle": "მართკუთხედი", "square": "კვადრატი",
    "quadrilateral": "ოთხკუთხედი", "polygon": "მრავალკუთხედი", "circle": "წრეწირი",
    "ellipse": "ელიფსი", "closed_curve": "ჩაკეტილი მრუდი", "line": "წრფის მონაკვეთი",
    "arc": "რკალი", "function": "ფუნქციის გრაფიკი", "freehand": "თავისუფალი მრუდი",
}

FAMILY_NAMES = {
    "linear": "წრფივი ფუნქცია", "quadratic": "პარაბოლა (კვადრატული)", "cubic": "კუბური ფუნქცია",
    "abs": "მოდულის ფუნქცია |x|", "sine": "სინუსოიდა", "sin": "სინუსი", "cos": "კოსინუსი",
    "sinusoid": "სინუსოიდა",
}


def tr(key: str, *args) -> str:
    text = T.get(key, key)
    return text.format(*args) if args else text


def kind_name(kind) -> str:
    return KIND_NAMES.get(getattr(kind, "value", kind), str(kind))
