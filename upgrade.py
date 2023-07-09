import wx
import os
import threading
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk

class TextureUpgrader(wx.Frame):
    def __init__(self, parent, title):
        super(TextureUpgrader, self).__init__(parent, title=title, size=(1000, 500))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("xinntao/ESRGAN", "RRDBNet_arch", pretrained=True)
        self.model = self.model.to(self.device).eval()
        self.preprocess = transforms.ToTensor()
        self.postprocess = transforms.ToPILImage()

        self.max_preview_width = 600
        self.max_preview_height = 400

        self.original_bitmap = None
        self.upgraded_bitmap = None

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

        progress_bar = wx.Gauge(panel, -1, style=wx.GA_HORIZONTAL)
        vbox.Add(progress_bar, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        select_button = wx.Button(panel, -1, "Select File")
        select_button.Bind(wx.EVT_BUTTON, self.on_select)
        vbox.Add(select_button, 0, wx.ALIGN_CENTER | wx.ALL, 10)

        panel.SetSizer(vbox)

    def on_select(self, event):
        dialog = wx.FileDialog(self, "Select a DDS file", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST, wildcard="DDS files (*.dds)|*.dds")
        if dialog.ShowModal() == wx.ID_OK:
            file_path = dialog.GetPath()
            self.upgrade_texture(file_path)
        dialog.Destroy()

    def upgrade_texture(self, file_path):
        img = Image.open(file_path).convert("RGB")
        img_width, img_height = img.size
        if img_width > self.max_preview_width or img_height > self.max_preview_height:
            img.thumbnail((self.max_preview_width, self.max_preview_height))

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

        output_img.save(upgraded_file_path)

        dlg = wx.MessageDialog(self, "Texture upgrade is complete.", "Upgrade Complete", wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

        self.original_bitmap = wx.BitmapFromImage(wx.Image(file_path, wx.BITMAP_TYPE_ANY))
        self.original_image_ctrl.SetBitmap(self.original_bitmap)

        self.upgraded_bitmap = wx.BitmapFromImage(wx.Image(upgraded_file_path, wx.BITMAP_TYPE_ANY))
        self.upgraded_image_ctrl.SetBitmap(self.upgraded_bitmap)

        self.display_image_info(file_path, upgraded_file_path)

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

app = wx.App()
TextureUpgrader(None, "Texture Upgrader")
app.MainLoop()
