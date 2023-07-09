import wx
import os
import threading
import torch
import imageio
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

        self.progress_bar = wx.Gauge(panel, -1, style=wx.GA_HORIZONTAL)
        vbox.Add(self.progress_bar, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        select_button = wx.Button(panel, -1, "Select Files")
        select_button.Bind(wx.EVT_BUTTON, self.on_select)
        vbox.Add(select_button, 0, wx.ALIGN_CENTER | wx.ALL, 10)

        self.SetSizer(vbox)

        self.file_paths = []

        self.menu_bar = wx.MenuBar()
        self.file_menu = wx.Menu()
        self.open_recent_menu = wx.Menu()
        self.save_menu = wx.Menu()

        self.file_menu.Append(wx.ID_OPEN, "&Open File...\tCtrl+O")
        self.file_menu.Append(wx.ID_SAVE, "&Save Upgraded Image...\tCtrl+S")
        self.file_menu.Append(wx.ID_EXIT, "E&xit")

        self.menu_bar.Append(self.file_menu, "&File")
        self.menu_bar.Append(self.open_recent_menu, "Open &Recent")
        self.menu_bar.Append(self.save_menu, "&Save")

        self.SetMenuBar(self.menu_bar)

        self.Bind(wx.EVT_MENU, self.on_open_file, id=wx.ID_OPEN)
        self.Bind(wx.EVT_MENU, self.on_save_upgraded_image, id=wx.ID_SAVE)
        self.Bind(wx.EVT_MENU, self.on_exit, id=wx.ID_EXIT)

    def on_select(self, event):
        dialog = wx.FileDialog(self, "Select DDS Files", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE, wildcard="DDS files (*.dds)|*.dds")
        if dialog.ShowModal() == wx.ID_OK:
            self.file_paths = dialog.GetPaths()
            self.upgrade_textures()
        dialog.Destroy()

    def upgrade_textures(self):
        self.progress_bar.SetRange(len(self.file_paths))
        self.progress_bar.SetValue(0)

        for index, file_path in enumerate(self.file_paths):
            thread = threading.Thread(target=self.upgrade_texture, args=(file_path, index))
            thread.start()

    def upgrade_texture(self, file_path, index):
        img = Image.open(file_path).convert("RGB")
        img_width, img_height = img.size
        if img_width > self.max_preview_width or img_height > self.max_preview_height:
            img.thumbnail((self.max_preview_width, self.max_preview_height))

        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)

        output_img = self.postprocess(output.squeeze(0).cpu())

        file_name, file_extension = os.path.splitext(file_path)
        upgraded_file_path = file_name + "_upgraded.dds"

        if os.path.exists(upgraded_file_path):
            dlg = wx.MessageDialog(self, "The upgraded file already exists. Do you want to overwrite it?", "File Exists", wx.YES_NO | wx.ICON_QUESTION)
            if dlg.ShowModal() == wx.ID_NO:
                return
            dlg.Destroy()

        output_img.save(upgraded_file_path)

        self.progress_bar.SetValue(index + 1)

        if index == len(self.file_paths) - 1:
            wx.CallAfter(self.display_completion_message)

    def display_completion_message(self):
        self.progress_bar.SetValue(0)
        dlg = wx.MessageDialog(self, "Texture upgrade is complete.", "Upgrade Complete", wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

    def on_open_file(self, event):
        self.on_select(event)

    def on_save_upgraded_image(self, event):
        dlg = wx.MessageDialog(self, "Saving back to the .dds format is not supported.", "Save Error", wx.OK | wx.ICON_ERROR)
        dlg.ShowModal()
        dlg.Destroy()

    def on_exit(self, event):
        self.Close()


if __name__ == "__main__":
    app = wx.App()
    TextureUpgrader(None, "Texture Upgrader")
    app.MainLoop()
