from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import ScreenManager, Screen


class JournalEntryWindow(Screen):
    journal = ObjectProperty(None)

    def submit(self):
        self.reset()
        sm.current = "selection"

    def reset(self):
        self.journal.text = ""


class MediaSelectionWindow(Screen):
    pass


class WindowManager(ScreenManager):
    pass


jkv = Builder.load_file("format.kv")

sm = WindowManager()

screens = [JournalEntryWindow(name="journal"), MediaSelectionWindow(name="selection")]
for screen in screens:
    sm.add_widget(screen)

sm.current = "journal"


class MyApp(App):
    def build(self):
        return sm


MyApp().run()
