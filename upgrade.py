import wx
import os
import threading
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import logging
import json

class TextureUpgrader(wx.Frame):
    def __init__(self, parent, title):
        super(TextureUpgrader, self).__init__(parent, title=title, size=(1000, 500))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("xinntao/ESRGAN", "RRDBNet_arch", pretrained=True)
        self.model = self.model.to(self.device).eval()
        self.preprocess = transforms.ToTensor()
        self.postprocess = transforms.ToPILImage()

        self.logger = self.setup_logger()  
        self.load_settings()  

        self.create_ui()

    def create_ui(self):
        panel = wx.Panel(self)

        vbox = wx.BoxSizer(wx.VERTICAL)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.original_image_ctrl = wx.StaticBitmap(panel, -1, wx.NullBitmap)
        hbox.Add(self.original_image_ctrl, 1, wx.EXPAND | wx.ALL, 10)
        self.upgraded_image_ctrl = wx.StaticBitmap(panel, -1, wx.NullBitmap)
        hbox.Add(self.upgraded_image_ctrl, 1, wx.EXPAND | wx.ALL, 10)

        vbox.Add(hbox, 1, wx.EXPAND)

        self.progress_bar = wx.Gauge(panel, -1, style=wx.GA_HORIZONTAL)
        vbox.Add(self.progress_bar, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        select_button = wx.Button(panel, -1, "Select File(s)")
        select_button.Bind(wx.EVT_BUTTON, self.on_select)
        vbox.Add(select_button, 0, wx.ALIGN_CENTER | wx.ALL, 10)

        convert_button = wx.Button(panel, -1, "Convert to DDS")
        convert_button.Bind(wx.EVT_BUTTON, self.on_convert)
        vbox.Add(convert_button, 0, wx.ALIGN_CENTER | wx.ALL, 10)

        self.thumbnail_ctrl = wx.ListCtrl(panel, -1, style=wx.LC_ICON)
        self.thumbnail_ctrl.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.on_thumbnail_selected)
        vbox.Add(self.thumbnail_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        panel.SetSizer(vbox)

        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_press)

def on_select(self, event):
    dialog = wx.FileDialog(self, "Select DDS file(s)", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE, wildcard="DDS files (*.dds)|*.dds")
    if dialog.ShowModal() == wx.ID_OK:
        file_paths = dialog.GetPaths()
        valid_files = self.validate_files(file_paths)
        if valid_files:
            self.upgrade_textures(valid_files)  
        else:
            wx.MessageBox("Invalid file(s) selected.", "Invalid Files", wx.OK | wx.ICON_ERROR)
    dialog.Destroy()

def on_convert(self, event):
    dialog = wx.FileDialog(self, "Select image file(s) to convert", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE)
    if dialog.ShowModal() == wx.ID_OK:
        file_paths = dialog.GetPaths()
        valid_files = self.validate_files(file_paths)
        if valid_files:
            self.convert_to_dds(valid_files)  
        else:
            wx.MessageBox("Invalid file(s) selected.", "Invalid Files", wx.OK | wx.ICON_ERROR)
    dialog.Destroy()

    def on_thumbnail_selected(self, event):
        index = event.GetIndex()
        selected_file = self.thumbnail_ctrl.GetItemData(index)
        file_path = self.thumbnail_files[selected_file]
        self.display_image(file_path)

    def on_key_press(self, event):
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_ESCAPE:  # Close application on ESC key
            self.Close()
        event.Skip()

    def upgrade_textures(self, file_paths):
        thread = threading.Thread(target=self.process_textures, args=(file_paths,))
        thread.start()

    def process_textures(self, file_paths):
        self.progress_bar.SetValue(0)
        self.progress_bar.Pulse()

        total_files = len(file_paths)
        successful_upgrades = 0
        failed_conversions = 0
        self.thumbnail_files = []

        for idx, file_path in enumerate(file_paths, start=1):
            self.logger.info(f"Processing texture {idx}/{total_files}: {file_path}")

            try:
                self.process_texture(file_path)
                successful_upgrades += 1
            except Exception as e:
                self.logger.error(f"Failed to upgrade texture: {file_path}")
                self.logger.exception(e)
                failed_conversions += 1

            if self.canceled:  
                self.logger.info("Texture upgrade canceled.")
                break

        self.logger.info("Texture upgrade batch operation completed.")
        self.logger.info(f"Total files: {total_files}")
        self.logger.info(f"Successful upgrades: {successful_upgrades}")
        self.logger.info(f"Failed conversions: {failed_conversions}")

        self.progress_bar.SetValue(0)
        self.canceled = False  

    def process_texture(self, file_path):
        self.progress_bar.Pulse()

        img = Image.open(file_path).convert("RGB")
        img_width, img_height = img.size

        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)

        output_img = self.postprocess(output.squeeze(0).cpu())

        file_name, file_extension = os.path.splitext(file_path)
        upgraded_file_path = file_name + "_upgraded" + file_extension

        if os.path.exists(upgraded_file_path):
            dlg = wx.MessageDialog(self, "The upgraded file already exists. Do you want to overwrite it?", "File Exists", wx.YES_NO | wx.ICON_QUESTION)
            if dlg.ShowModal() == wx.ID_NO:
                return
            dlg.Destroy()

        output_img = output_img.resize((img_width, img_height), resample=Image.LANCZOS)

        output_img.save(upgraded_file_path, format="DDS")

        self.update_image_preview(file_path, upgraded_file_path)
        self.display_image_info(file_path, upgraded_file_path)

        self.thumbnail_files.append(upgraded_file_path)
        self.add_thumbnail(upgraded_file_path)

    def convert_to_dds(self, file_paths):
        for file_path in file_paths:
            file_name, file_extension = os.path.splitext(file_path)
            converted_file_path = file_name + ".dds"

            if os.path.exists(converted_file_path):
                dlg = wx.MessageDialog(self, "The converted file already exists. Do you want to overwrite it?", "File Exists", wx.YES_NO | wx.ICON_QUESTION)
                if dlg.ShowModal() == wx.ID_NO:
                    continue
                dlg.Destroy()

            img = Image.open(file_path)

            img.save(converted_file_path, format="DDS")

            self.thumbnail_files.append(converted_file_path)
            self.add_thumbnail(converted_file_path)

    def add_thumbnail(self, file_path):
        image = wx.Image(file_path, wx.BITMAP_TYPE_ANY)
        thumbnail = image.Scale(128, 128, wx.IMAGE_QUALITY_HIGH)
        thumbnail_bitmap = thumbnail.ConvertToBitmap()
        image_index = self.thumbnail_ctrl.GetItemCount()

        self.thumbnail_ctrl.InsertItem(image_index, file_path, image_index)
        self.thumbnail_ctrl.SetItemImage(image_index, thumbnail_bitmap)
        self.thumbnail_ctrl.SetItemData(image_index, image_index)

    def update_image_preview(self, original_path, upgraded_path):
        self.original_image_ctrl.SetBitmap(wx.BitmapFromImage(wx.Image(original_path, wx.BITMAP_TYPE_ANY)))
        self.upgraded_image_ctrl.SetBitmap(wx.BitmapFromImage(wx.Image(upgraded_path, wx.BITMAP_TYPE_ANY)))

    def display_image_info(self, original_path, upgraded_path):
        original_info = self.get_image_info(original_path)
        upgraded_info = self.get_image_info(upgraded_path)

        dlg = wx.MessageDialog(self, f"Original Image:\n\n{original_info}\n\nUpgraded Image:\n\n{upgraded_info}", "Image Information", wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

    def get_image_info(self, file_path):
        image = Image.open(file_path)
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        width, height = image.size
        mode = image.mode

        return f"File Name: {file_name}\nFile Size: {file_size} bytes\nDimensions: {width} x {height}\nMode: {mode}"

    def setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        log_file = os.path.join(os.getcwd(), "texture_upgrader.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        return logger

    def validate_files(self, file_paths):
    valid_files = []
    for file_path in file_paths:
        if self.is_valid_image(file_path):
            valid_files.append(file_path)
        else:
            self.logger.warning(f"Invalid file: {file_path}")
    return valid_files

def is_valid_image(self, file_path):
    try:
        image = Image.open(file_path)
        image.verify()  
        return True
    except (IOError, SyntaxError) as e:
        self.logger.exception(e)
        return False

    def load_settings(self):
        settings_file = os.path.join(os.getcwd(), "texture_upgrader_settings.json")
        if os.path.exists(settings_file):
            with open(settings_file, "r") as f:
                settings = json.load(f)

    def save_settings(self):
        settings_file = os.path.join(os.getcwd(), "texture_upgrader_settings.json")
        settings = {
        }
        with open(settings_file, "w") as f:
            json.dump(settings, f)

app = wx.App()
TextureUpgrader(None, "Texture Upgrader")
app.MainLoop()
