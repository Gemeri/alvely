import sys
import os
import requests
import re
import markdown
import base64
from pdf2image import convert_from_path
import os
from dotenv import load_dotenv

load_dotenv()

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QScrollArea, QPushButton, QFrame, QStackedLayout, QSizePolicy,
    QShortcut, QMenu, QTextBrowser, QFileDialog, QComboBox, QSpacerItem,
    QLayout, QGridLayout, QMainWindow, QTabWidget, QToolButton
)
from PyQt5.QtCore import (
    Qt, pyqtSignal, QObject, QThread, QTimer, QRegExp, QEvent, QSize, QRect, QPoint
)
from PyQt5.QtGui import (
    QFont, QPixmap, QIcon, QTransform, QTextCharFormat, QKeySequence, QTextCursor, QContextMenuEvent
)
from openai import OpenAI
import anthropic
from bs4 import BeautifulSoup
from urllib.parse import urlparse

###############################################################################
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and PyInstaller """
    try:
        base_path = sys._MEIPASS  # type: ignore
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

###############################################################################
class Worker(QObject):
    result_ready = pyqtSignal(str)
    sources_ready = pyqtSignal(list)
    images_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self, query, conversation_history, client, anthropic_client,
        bing_api_key, uploaded_files, mode, model_id,
        fetched_urls=None,
        fetched_image_urls=None
    ):
        super().__init__()
        self.query = query
        self.conversation_history = conversation_history.copy()
        self.client = client
        self.anthropic_client = anthropic_client
        self.bing_api_key = bing_api_key
        self.uploaded_files = uploaded_files.copy()
        self.mode = mode
        self.model_id = model_id
        self.fetched_urls = fetched_urls if fetched_urls is not None else set()
        self.fetched_image_urls = fetched_image_urls if fetched_image_urls is not None else set()

    def run(self):
        try:
            if self.mode == 'text':
                # If there's already an assistant answer, we do the "More" approach:
                if any(msg['role'] == 'assistant' for msg in self.conversation_history):
                    # Just fetch more links from the user’s single typed query, no AI
                    new_links = self.fetchMoreLinks(self.query)
                    self.sources_ready.emit(new_links)
                else:
                    # Normal approach: get related queries => search => AI summarization
                    related = self.getRelatedQueries(self.query)
                    all_links = self.getSearchResults(related)
                    new_links = []
                    for link in all_links:
                        if link['url'] not in self.fetched_urls:
                            self.fetched_urls.add(link['url'])
                            new_links.append(link)
                    content_map = self.getWebsiteContents(new_links)
                    ai_answer = self.generateResponse(self.query, content_map)
                    self.result_ready.emit(ai_answer)
                    self.sources_ready.emit(new_links)

            elif self.mode == 'image':
                # Always do AI-based approach => skip duplicates
                related = self.getRelatedQueries(self.query)
                all_imgs = self.getImageResults(related)
                new_images = []
                for img in all_imgs:
                    if img['thumbnailUrl'] not in self.fetched_image_urls:
                        self.fetched_image_urls.add(img['thumbnailUrl'])
                        new_images.append(img)
                self.images_ready.emit(new_images)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.finished.emit()

    def fetchMoreLinks(self, query):
        # Single typed query, skip duplicates
        new_links = []
        results = self.bing_web_search(query)
        for r in results:
            if r['url'] not in self.fetched_urls:
                self.fetched_urls.add(r['url'])
                new_links.append(r)
        return new_links

    def getRelatedQueries(self, query):
        prompt_text = (
            f"Generate a list of detailed search queries that expand upon the topic: '{query}'. "
            "Provide each query on a new line."
        )
        messages = [{"role": "system", "content": "You are an assistant that generates related search queries."}]

        if self.uploaded_files:
            user_content = [{"type": "text", "text": prompt_text}]
            for file in self.uploaded_files:
                if file['type'] == 'image':
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{file['data']}"}
                    })
                else:
                    user_content.append({
                        "type": "text",
                        "text": f"Additional context from uploaded {file['type']} file:\n{file['data']}"
                    })
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": prompt_text})

        if self.model_id in ['gpt-4o', 'gpt-4o-mini', 'o1-mini', 'o1-preview']:
            comp = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages
            )
            return [x.strip('- ').strip() for x in comp.choices[0].message.content.split('\n') if x.strip()]
        else:
            anthro_resp = self.anthropic_client.completions.create(
                model=self.model_id,
                max_tokens_to_sample=1024,
                prompt=anthropic.HUMAN_PROMPT + prompt_text + anthropic.AI_PROMPT
            )
            lines = anthro_resp.completion.strip().split('\n')
            return [x.strip('- ').strip() for x in lines if x.strip()]

    def getSearchResults(self, queries):
        results = []
        for q in queries:
            results.extend(self.bing_web_search(q))
        return results

    def bing_web_search(self, query):
        search_url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}
        params = {"q": query, "textDecorations": True, "textFormat": "HTML", "count": 2}
        r = requests.get(search_url, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()
        found = []
        if 'webPages' in data:
            for v in data['webPages']['value']:
                found.append({
                    'name': v['name'],
                    'url': v['url'],
                    'snippet': v['snippet'],
                    'displayUrl': v['displayUrl']
                })
        return found

    def getWebsiteContents(self, search_results):
        out = {}
        for item in search_results:
            out[item['url']] = f"{item['name']}: {item['snippet']}"
        return out

    def generateResponse(self, query, website_contents):
        sources = "\n".join([f"{txt} (Source: {url})" for url, txt in website_contents.items()])
        prompt_text = (
            f"Using the following information from various sources, answer the query: \"{query}\" "
            "and cite the sources in your response.\n\n"
            f"Information:\n{sources}\n\n"
            "Provide a detailed answer, and include the URLs of the sources you used."
        )
        messages = self.conversation_history.copy()

        if self.uploaded_files:
            user_content = [{"type": "text", "text": prompt_text}]
            for file in self.uploaded_files:
                if file['type'] == 'image':
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{file['data']}"}
                    })
                else:
                    user_content.append({
                        "type": "text",
                        "text": f"Additional context from uploaded {file['type']} file:\n{file['data']}"
                    })
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": prompt_text})

        if self.model_id in ['gpt-4o', 'gpt-4o-mini', 'o1-mini', 'o1-preview']:
            comp = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages
            )
            return comp.choices[0].message.content
        else:
            anthro_resp = self.anthropic_client.completions.create(
                model=self.model_id,
                max_tokens_to_sample=1024,
                prompt=anthropic.HUMAN_PROMPT + prompt_text + anthropic.AI_PROMPT
            )
            return anthro_resp.completion.strip()

    def getImageResults(self, queries):
        # For images, each query => 10 images
        results = []
        for q in queries:
            search_url = "https://api.bing.microsoft.com/v7.0/images/search"
            headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}
            params = {"q": q, "count": 10, "imageType": "photo"}
            resp = requests.get(search_url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            if 'value' in data:
                for v in data['value']:
                    results.append({
                        'thumbnailUrl': v.get('thumbnailUrl', ''),
                        'contentUrl': v.get('contentUrl', ''),
                        'hostPageUrl': v.get('hostPageUrl', '')
                    })
        return results

###############################################################################
class CopyableLabel(QLabel):
    def __init__(self, text='', parent=None):
        super().__init__(text, parent)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)

    def contextMenuEvent(self, event: QContextMenuEvent):
        menu = QMenu(self)
        copy_action = menu.addAction("Copy")
        action = menu.exec_(self.mapToGlobal(event.pos()))
        if action == copy_action:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.text())

###############################################################################
class MessageWidget(QWidget):
    def __init__(self, sender, message, mode='text'):
        super().__init__()
        self.sender = sender
        self.message = message
        self.mode = mode
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        sender_label = QLabel(f'{self.sender}:')
        sender_label.setFont(QFont('Arial', 10, QFont.Bold))
        layout.addWidget(sender_label)

        if self.sender == 'Assistant' and self.mode == 'text':
            self.message_display = QTextBrowser()
            self.message_display.setReadOnly(True)
            self.message_display.setOpenExternalLinks(True)
            self.message_display.setFont(QFont('Arial', 12))
            self.message_display.setStyleSheet(
                "QTextBrowser { border: none; background-color: transparent; }"
            )

            html_content = self.processMessage(self.message)
            self.message_display.setHtml(html_content)
            self.message_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            self.message_display.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.message_display.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            layout.addWidget(self.message_display)

            self.message_display.document().setTextWidth(self.message_display.viewport().width())
            height = self.message_display.document().size().height()
            self.message_display.setFixedHeight(int(height) + 10)

            self.copy_button = QPushButton('Copy Response')
            self.copy_button.clicked.connect(self.copyResponse)
            self.copy_button.setFixedWidth(120)
            layout.addWidget(self.copy_button, alignment=Qt.AlignRight)

        else:
            self.message_display = QLabel(self.message)
            self.message_display.setFont(QFont('Arial', 12))
            self.message_display.setWordWrap(True)
            layout.addWidget(self.message_display)

    def processMessage(self, message):
        html = markdown.markdown(message, extensions=['fenced_code', 'tables'])
        html = re.sub(r'\\\((.*?)\\\)', r'<i>\1</i>', html)
        html = re.sub(r'\\\[(.*?)\\\]', r'<i>\1</i>', html)
        return html

    def copyResponse(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.message)
        self.copy_button.setText("Copied")
        QTimer.singleShot(5000, self.resetCopyButton)

    def resetCopyButton(self):
        self.copy_button.setText("Copy Response")

    def highlightText(self, text):
        if hasattr(self, 'message_display') and isinstance(self.message_display, QTextBrowser):
            cursor = self.message_display.textCursor()
            fmt = QTextCharFormat()
            fmt.setBackground(Qt.yellow)

            regex = QRegExp(text, Qt.CaseInsensitive)
            cursor.beginEditBlock()

            cursor.select(QTextCursor.Document)
            cursor.setCharFormat(QTextCharFormat())

            matches = []
            pos = 0
            doc_text = self.message_display.toPlainText()
            while True:
                pos = regex.indexIn(doc_text, pos)
                if pos < 0:
                    break
                cursor.setPosition(pos)
                cursor.movePosition(
                    QTextCursor.NextCharacter,
                    QTextCursor.KeepAnchor,
                    regex.matchedLength()
                )
                cursor.mergeCharFormat(fmt)
                matches.append(pos)
                pos += regex.matchedLength()

            cursor.endEditBlock()
            return matches
        return []

    def clearHighlights(self):
        if hasattr(self, 'message_display') and isinstance(self.message_display, QTextBrowser):
            cursor = self.message_display.textCursor()
            cursor.select(QTextCursor.Document)
            cursor.setCharFormat(QTextCharFormat())

###############################################################################
class ImageWidget(QWidget):
    def __init__(self, image_url, link_url, parent=None):
        super().__init__(parent)
        self.image_url = image_url
        self.link_url = link_url
        self.initUI()

    def initUI(self):
        # We want 2 images per row in a bigger box, say ~300 px wide
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        try:
            resp = requests.get(self.image_url)
            resp.raise_for_status()
            pixmap = QPixmap()
            pixmap.loadFromData(resp.content)
            # Scale to ~300 px wide to ensure we only get 2 images per row
            self.image_label.setPixmap(
                pixmap.scaledToWidth(300, Qt.SmoothTransformation)
            )
        except:
            self.image_label.setText("Image not available")
        layout.addWidget(self.image_label)

        self.link_label = QLabel(f"<a href='{self.link_url}' style='color: #55AAFF;'>View Source</a>")
        self.link_label.setAlignment(Qt.AlignCenter)
        self.link_label.setOpenExternalLinks(True)
        layout.addWidget(self.link_label)

    def contextMenuEvent(self, event: QContextMenuEvent):
        menu = QMenu(self)
        download_action = menu.addAction("Download Image")
        action = menu.exec_(self.mapToGlobal(event.pos()))
        if action == download_action:
            self.downloadImage()

    def downloadImage(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)",
            options=options
        )
        if file_path:
            try:
                resp = requests.get(self.image_url)
                resp.raise_for_status()
                with open(file_path, 'wb') as f:
                    f.write(resp.content)
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Download Error", f"Failed to download image: {str(e)}")

###############################################################################
class SourceWidget(QWidget):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.initUI()

    def initUI(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #3a3a3a;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        icon_label = QLabel()
        icon_pixmap = self.getFavicon(self.source['url'])
        icon_label.setPixmap(
            icon_pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        icon_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        icon_label.setMaximumSize(64, 64)
        layout.addWidget(icon_label, 0, Qt.AlignLeft)

        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(2)

        title_label = CopyableLabel(self.source['name'])
        title_label.setFont(QFont('Arial', 12, QFont.Bold))
        title_label.setWordWrap(True)
        info_layout.addWidget(title_label)

        url_label = QLabel(
            f"<a href='{self.source['url']}' style='color: #55AAFF;'>{self.source['displayUrl']}</a>"
        )
        url_label.setFont(QFont('Arial', 10))
        url_label.setStyleSheet("QLabel { color: #55AAFF; }")
        url_label.setOpenExternalLinks(True)
        info_layout.addWidget(url_label)

        info_widget = QWidget()
        info_widget.setLayout(info_layout)
        info_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(info_widget)
        layout.setStretch(1, 1)
        layout.setAlignment(Qt.AlignLeft)

    def getFavicon(self, url):
        try:
            domain = urlparse(url).netloc
            fav_url = f"https://www.google.com/s2/favicons?sz=64&domain_url={domain}"
            r = requests.get(fav_url)
            pix = QPixmap()
            pix.loadFromData(r.content)
            return pix
        except:
            return QPixmap(resource_path('assets/default_icon.png'))

###############################################################################
class UploadedFilesWidget(QFrame):
    def __init__(self, parent=None, max_width=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #333333;
                border-radius: 10px;
            }
        """)
        self.setFixedHeight(80)
        if max_width:
            self.setMaximumWidth(max_width)
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        else:
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setVisible(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; }")
        self.scroll_area.setFixedHeight(60)

        self.container = QWidget()
        self.container_layout = QHBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(10)

        self.scroll_area.setWidget(self.container)
        layout.addWidget(self.scroll_area)

    def addFile(self, file_type, file_name):
        file_widget = QWidget()
        fw_layout = QVBoxLayout(file_widget)
        fw_layout.setContentsMargins(5, 5, 5, 5)
        fw_layout.setAlignment(Qt.AlignCenter)
        fw_layout.setSpacing(5)

        if file_type == 'text':
            symbol_path = resource_path('assets/text_icon.png')
        elif file_type == 'code':
            symbol_path = resource_path('assets/code_icon.png')
        elif file_type == 'image':
            symbol_path = resource_path('assets/image_icon.png')
        elif file_type == 'pdf':
            symbol_path = resource_path('assets/pdf_icon.png')
        else:
            symbol_path = resource_path('assets/file_icon.png')

        symbol_label = QLabel()
        symbol_pix = QPixmap(symbol_path).scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        symbol_label.setPixmap(symbol_pix)
        symbol_label.setAlignment(Qt.AlignCenter)
        fw_layout.addWidget(symbol_label)

        fn_label = QLabel(file_name)
        fn_label.setFont(QFont('Arial', 10))
        fn_label.setAlignment(Qt.AlignCenter)
        fn_label.setStyleSheet("color: white;")
        fn_label.setWordWrap(True)
        fw_layout.addWidget(fn_label)

        self.container_layout.addWidget(file_widget)
        self.setVisible(True)

    def clearFiles(self):
        while self.container_layout.count():
            item = self.container_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.setVisible(False)

###############################################################################
class ChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.current_mode = 'text'
        self.selected_model = 'gpt-4o-mini'
        self.conversation_history = []
        self.uploaded_files = []
        self.bing_api_key = os.getenv("BING_API_KEY", "")

        self.source_widgets = []
        self.image_widgets = []
        self.worker_thread = None
        self.image_offset = 0

        # Track duplicates
        self.fetched_urls = set()
        self.fetched_image_urls = set()

        self.initUI()
        self.setupClients()

    def initUI(self):
        self.setWindowTitle('Alvely')
        self.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
                color: white;
            }
            QLineEdit {
                background-color: #333333;
                color: white;
                padding: 10px;
                border-radius: 20px;
                font-size: 16px;
            }
            QPushButton {
                background-color: #333333;
                color: white;
                padding: 5px;
                border-radius: 5px;
            }
            QLabel {
                color: white;
            }
            QComboBox {
                background-color: #333333;
                color: white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        self.resize(1000, 700)

        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)

        # Top bar (60px tall)
        self.top_bar = QFrame()
        self.top_bar.setStyleSheet("QFrame { background-color: #2A2A2A; }")
        self.top_bar.setFixedHeight(60)

        # Use a QGridLayout for precise 3-column control
        top_layout = QGridLayout(self.top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)

        # Column stretching: columns 0 & 2 expand equally, so column 1 is center
        top_layout.setColumnStretch(0, 1)  # Left
        top_layout.setColumnStretch(1, 0)  # Center
        top_layout.setColumnStretch(2, 1)  # Right

        # 1) Model dropdown => column 0, aligned left
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems([
            "Fast (gpt-4o-mini)",
            "Faster (claude-3-5-haiku-latest)",
            "Fastest (o1-mini)",
            "Smart (gpt-4o)",
            "Smarter (claude-3-5-sonnet-latest)",
            "Smartest (o1-preview)"
        ])
        self.model_dropdown.currentIndexChanged.connect(self.changeModel)
        top_layout.addWidget(self.model_dropdown, 0, 0, Qt.AlignLeft)

        # 2) Center widget => column 1, aligned center
        center_widget = QWidget()
        center_layout = QHBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(8)

        self.text_button = QPushButton("Text")
        self.text_button.setCheckable(True)
        self.text_button.setChecked(self.current_mode == 'text')
        self.text_button.clicked.connect(self.setTextMode)
        center_layout.addWidget(self.text_button)

        separator = QLabel("|")
        separator.setFont(QFont('Arial', 16))
        separator.setStyleSheet("color: white;")
        center_layout.addWidget(separator)

        self.image_button = QPushButton("Image")
        self.image_button.setCheckable(True)
        self.image_button.setChecked(self.current_mode == 'image')
        self.image_button.clicked.connect(self.setImageMode)
        center_layout.addWidget(self.image_button)

        top_layout.addWidget(center_widget, 0, 1, Qt.AlignCenter)

        # 3) Settings => column 2, aligned right
        settings_button = QPushButton()
        settings_icon = QIcon(resource_path('assets/settings.png'))
        settings_button.setIcon(settings_icon)
        settings_button.setIconSize(QSize(24, 24))
        settings_button.setStyleSheet("QPushButton { background-color: transparent; }")
        settings_button.clicked.connect(self.toggleSettingsPanel)
        top_layout.addWidget(settings_button, 0, 2, Qt.AlignRight)

        self.main_layout.addWidget(self.top_bar)

        # -------------- Everything below is unchanged (initial page, chat page, etc.) --------------

        self.stack = QStackedLayout()
        self.main_layout.addLayout(self.stack)

        # Initial page
        self.init_page = QWidget()
        self.init_layout = QVBoxLayout(self.init_page)
        self.init_layout.setAlignment(Qt.AlignCenter)

        self.logo_label_large = QLabel()
        logo_pixmap_large = QPixmap(resource_path('assets/Alvely.png')).scaled(
            150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.logo_label_large.setPixmap(logo_pixmap_large)
        self.logo_label_large.setAlignment(Qt.AlignCenter)
        self.init_layout.addWidget(self.logo_label_large)

        self.title_label_large = QLabel('Alvely')
        self.title_label_large.setFont(QFont('Arial', 48, QFont.Bold))
        self.title_label_large.setAlignment(Qt.AlignCenter)
        self.init_layout.addWidget(self.title_label_large)

        # Row with file button + search bar
        search_layout = QHBoxLayout()
        search_layout.setSpacing(10)
        search_layout.setContentsMargins(0, 0, 0, 0)

        self.init_file_button = QPushButton()
        file_icon = QIcon(resource_path('assets/file.png'))
        self.init_file_button.setIcon(file_icon)
        self.init_file_button.setIconSize(QSize(24, 24))
        self.init_file_button.setFixedSize(40, 40)
        self.init_file_button.setStyleSheet("QPushButton { background-color: #333333; border-radius: 20px; }")
        self.init_file_button.clicked.connect(self.uploadFiles)
        search_layout.addWidget(self.init_file_button)

        self.init_search_bar = QLineEdit()
        self.init_search_bar.setPlaceholderText('Type your query here...')
        self.init_search_bar.returnPressed.connect(self.onFirstSubmit)
        self.init_search_bar.setFixedWidth(400)
        self.init_search_bar.setFixedHeight(40)
        self.init_search_bar.setAlignment(Qt.AlignLeft)
        self.init_search_bar.setFont(QFont('Arial', 14))
        search_layout.addWidget(self.init_search_bar)

        self.init_layout.addLayout(search_layout)
        self.init_uploaded_files_widget = UploadedFilesWidget(max_width=400)
        self.init_layout.addWidget(self.init_uploaded_files_widget)

        self.stack.addWidget(self.init_page)

        # Chat page
        self.chat_page = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_page)
        self.chat_layout.setContentsMargins(0, 0, 0, 0)
        self.chat_layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; }")

        self.scroll_content = QWidget()
        self.scroll_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_layout.setContentsMargins(20, 10, 20, 10)
        self.scroll_layout.setSpacing(10)
        self.scroll_area.setWidget(self.scroll_content)

        self.chat_layout.addWidget(self.scroll_area)

        self.chat_uploaded_files_widget = UploadedFilesWidget()
        self.chat_layout.addWidget(self.chat_uploaded_files_widget)

        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(20, 10, 20, 20)
        input_layout.setSpacing(10)

        self.chat_file_button = QPushButton()
        file_icon_chat = QIcon(resource_path('assets/file.png'))
        self.chat_file_button.setIcon(file_icon_chat)
        self.chat_file_button.setIconSize(QSize(24, 24))
        self.chat_file_button.setFixedSize(40, 40)
        self.chat_file_button.setStyleSheet("QPushButton { background-color: #333333; border-radius: 20px; }")
        self.chat_file_button.clicked.connect(self.uploadFiles)
        input_layout.addWidget(self.chat_file_button)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText('Type your query here...')
        self.input_field.returnPressed.connect(self.onSubmit)
        self.input_field.setFixedHeight(40)
        self.input_field.setFont(QFont('Arial', 14))
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #333333;
                color: white;
                padding: 10px;
                border-radius: 20px;
                font-size: 16px;
            }
        """)
        input_layout.addWidget(self.input_field)
        self.chat_layout.addLayout(input_layout)

        self.stack.addWidget(self.chat_page)

        # Loading + Error overlays
        self.loading_widget = QLabel()
        self.loading_widget.setAlignment(Qt.AlignCenter)
        self.loading_pixmap = QPixmap(resource_path('assets/Alvely.png')).scaled(
            100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.loading_widget.setPixmap(self.loading_pixmap)
        self.loading_widget.hide()
        self.rotation_angle = 0
        self.loading_timer = QTimer()
        self.loading_timer.timeout.connect(self.rotateLoadingImage)

        self.error_widget = QLabel()
        self.error_widget.setAlignment(Qt.AlignCenter)
        self.error_widget.hide()

        self.overlay_layout = QVBoxLayout()
        self.overlay_layout.setAlignment(Qt.AlignCenter)
        self.main_layout.addLayout(self.overlay_layout)
        self.overlay_layout.addWidget(self.loading_widget)
        self.overlay_layout.addWidget(self.error_widget)

        self.settings_panel = None
        self.find_widget = None
        QShortcut(QKeySequence("Ctrl+F"), self, self.showFindDialog)

        self.uploaded_files_widgets = {
            'init': self.init_uploaded_files_widget,
            'chat': self.chat_uploaded_files_widget
        }


    def setupClients(self):
        print("[DEBUG] Setting up OpenAI and Anthropc clients.")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=self.openai_api_key)

        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)

    def getModeButtonStyle(self, mode):
        if self.current_mode == mode:
            return """
                QPushButton {
                    background-color: #555555;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 10px;
                }
            """
        else:
            return """
                QPushButton {
                    background-color: #333333;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 10px;
                }
            """

    def setTextMode(self):
        if self.current_mode != 'text':
            print("[DEBUG] Switching to text mode.")
            self.current_mode = 'text'
            self.text_button.setChecked(True)
            self.image_button.setChecked(False)
            self.text_button.setStyleSheet(self.getModeButtonStyle('text'))
            self.image_button.setStyleSheet(self.getModeButtonStyle('image'))
            # Return to initial page
            self.stack.setCurrentWidget(self.init_page)
            self.clearResults()

    def setImageMode(self):
        if self.current_mode != 'image':
            print("[DEBUG] Switching to image mode.")
            self.current_mode = 'image'
            self.text_button.setChecked(False)
            self.image_button.setChecked(True)
            self.text_button.setStyleSheet(self.getModeButtonStyle('text'))
            self.image_button.setStyleSheet(self.getModeButtonStyle('image'))
            # Return to initial page
            self.stack.setCurrentWidget(self.init_page)
            self.clearResults()

    def changeModel(self):
        selected_text = self.model_dropdown.currentText()
        model_id = selected_text.split('(')[-1].strip(')')
        print(f"[DEBUG] Changing model to: {model_id}")
        self.selected_model = model_id
        self.clearUploads()
        self.updateFileUploadAvailability()

    def updateFileUploadAvailability(self):
        self.init_file_button.setEnabled(True)
        self.chat_file_button.setEnabled(True)
        self.init_file_button.setStyleSheet("QPushButton { background-color: #333333; border-radius: 20px; }")
        self.chat_file_button.setStyleSheet("QPushButton { background-color: #333333; border-radius: 20px; }")

    def clearUploads(self):
        print("[DEBUG] Clearing uploaded files.")
        self.uploaded_files.clear()
        self.init_uploaded_files_widget.clearFiles()
        self.chat_uploaded_files_widget.clearFiles()
        self.clearResults()

    def clearResults(self):
        print("[DEBUG] Clearing chat results.")
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            if item:
                w = item.widget()
                if w:
                    w.deleteLater()
        self.source_widgets.clear()
        self.image_widgets.clear()
        self.image_offset = 0
        # Clear duplicates
        self.fetched_urls.clear()
        self.fetched_image_urls.clear()

    def rotateLoadingImage(self):
        self.rotation_angle = (self.rotation_angle + 10) % 360
        transform = QTransform().rotate(self.rotation_angle)
        rp = self.loading_pixmap.transformed(transform, Qt.SmoothTransformation)
        self.loading_widget.setPixmap(rp)

    def showLoading(self):
        print("[DEBUG] showLoading called.")
        self.loading_widget.show()
        self.loading_timer.start(50)

    def hideLoading(self):
        print("[DEBUG] hideLoading called.")
        self.loading_timer.stop()
        self.loading_widget.hide()
        self.rotation_angle = 0

    def showError(self, error_message):
        print("[DEBUG] showError called:", error_message)
        self.error_widget.show()
        err_layout = QVBoxLayout()
        self.error_widget.setLayout(err_layout)

        err_icon_label = QLabel()
        err_icon_pix = QPixmap(resource_path('assets/!.png')).scaled(
            100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        err_icon_label.setPixmap(err_icon_pix)
        err_icon_label.setAlignment(Qt.AlignCenter)
        err_layout.addWidget(err_icon_label)

        err_bg = QFrame()
        err_bg.setStyleSheet("QFrame { background-color: red; border-radius: 10px; }")
        err_bg_layout = QVBoxLayout(err_bg)
        err_bg_layout.setAlignment(Qt.AlignCenter)

        err_title = QLabel("Error")
        err_title.setFont(QFont('Arial', 16, QFont.Bold))
        err_title.setStyleSheet("QLabel { color: white; }")
        err_bg_layout.addWidget(err_title)

        err_detail = QLabel(error_message)
        err_detail.setStyleSheet("QLabel { color: white; }")
        err_detail.setWordWrap(True)
        err_bg_layout.addWidget(err_detail)

        err_layout.addWidget(err_bg)
        QTimer.singleShot(5000, self.hideError)

    def hideError(self):
        self.error_widget.hide()

    def onFirstSubmit(self):
        query = self.init_search_bar.text().strip()
        if not query:
            return
        self.init_search_bar.clear()
        self.stack.setCurrentWidget(self.chat_page)

        self.conversation_history.append({'role': 'user', 'content': query})
        user_msg = MessageWidget('User', query, mode=self.current_mode)
        self.scroll_layout.addWidget(user_msg)
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

        for w in self.source_widgets:
            self.scroll_layout.removeWidget(w)
            w.deleteLater()
        self.source_widgets.clear()

        for w in self.image_widgets:
            self.scroll_layout.removeWidget(w)
            w.deleteLater()
        self.image_widgets.clear()

        self.init_uploaded_files_widget.clearFiles()
        self.chat_uploaded_files_widget.clearFiles()
        self.uploaded_files.clear()
        self.startWorker(query)

    def onSubmit(self):
        query = self.input_field.text().strip()
        if not query:
            return
        self.input_field.clear()
        self.conversation_history.append({'role': 'user', 'content': query})
        user_msg = MessageWidget('User', query, mode=self.current_mode)
        self.scroll_layout.addWidget(user_msg)
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

        for w in self.source_widgets:
            self.scroll_layout.removeWidget(w)
            w.deleteLater()
        self.source_widgets.clear()

        for w in self.image_widgets:
            self.scroll_layout.removeWidget(w)
            w.deleteLater()
        self.image_widgets.clear()

        self.init_uploaded_files_widget.clearFiles()
        self.chat_uploaded_files_widget.clearFiles()
        self.uploaded_files.clear()
        self.startWorker(query)

    def startWorker(self, query):
        print(f"[DEBUG] startWorker called with query='{query}', mode='{self.current_mode}', model='{self.selected_model}'.")
        self.showLoading()
        self.worker_thread = QThread()
        self.worker = Worker(
            query=query,
            conversation_history=self.conversation_history,
            client=self.client,
            anthropic_client=self.anthropic_client,
            bing_api_key=self.bing_api_key,
            uploaded_files=self.uploaded_files,
            mode=self.current_mode,
            model_id=self.selected_model,
            fetched_urls=self.fetched_urls,
            fetched_image_urls=self.fetched_image_urls
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)

        self.worker.result_ready.connect(self.handleResult)
        self.worker.sources_ready.connect(self.displaySources)
        self.worker.images_ready.connect(self.displayImages)
        self.worker.error_occurred.connect(self.handleError)

        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def handleResult(self, result):
        print("[DEBUG] handleResult called.")
        self.hideLoading()
        self.conversation_history.append({'role': 'assistant', 'content': result})
        msg = MessageWidget('Assistant', result, mode=self.current_mode)
        self.scroll_layout.addWidget(msg)
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

        if self.current_mode == 'text':
            more_button = QPushButton('More')
            more_button.setFixedWidth(80)
            more_button.clicked.connect(self.loadMoreResults)
            self.scroll_layout.addWidget(more_button, alignment=Qt.AlignCenter)

    def handleError(self, error_message):
        print("[DEBUG] handleError called with:", error_message)
        self.hideLoading()
        self.showError(error_message)

    def displaySources(self, search_results):
        # Called after we fetch text links
        print("[DEBUG] displaySources called with", len(search_results), "results.")
        self.hideLoading()
        for item in search_results:
            sw = SourceWidget(item)
            self.scroll_layout.addWidget(sw, alignment=Qt.AlignLeft)
            self.source_widgets.append(sw)
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    def displayImages(self, image_results):
        # Show images in a 2-column grid
        print("[DEBUG] displayImages called with", len(image_results), "images.")
        self.hideLoading()

        # We'll build a QGridLayout: 2 columns, many rows
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(10)

        row = 0
        col = 0
        for idx, img in enumerate(image_results):
            iw = ImageWidget(img['thumbnailUrl'], img['hostPageUrl'])
            self.image_widgets.append(iw)

            grid_layout.addWidget(iw, row, col, Qt.AlignCenter)
            col += 1
            if col >= 2:
                row += 1
                col = 0

        self.scroll_layout.addWidget(grid_widget, alignment=Qt.AlignLeft)

        more_button = QPushButton('More')
        more_button.setFixedWidth(80)
        more_button.clicked.connect(self.loadMoreImages)
        self.scroll_layout.addWidget(more_button, alignment=Qt.AlignCenter)

        self.image_offset += len(image_results)
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    def loadMoreResults(self):
        print("[DEBUG] loadMoreResults called.")
        # Single typed query only, skip duplicates
        user_query = None
        for msg in reversed(self.conversation_history):
            if msg['role'] == 'user':
                user_query = msg['content']
                break
        if user_query:
            self.startWorker(user_query)

    def loadMoreImages(self):
        print("[DEBUG] loadMoreImages called.")
        # AI approach each time, skip duplicates
        user_query = None
        for msg in reversed(self.conversation_history):
            if msg['role'] == 'user':
                user_query = msg['content']
                break
        if user_query:
            self.startWorker(user_query)

    def reloadApp(self):
        print("[DEBUG] reloadApp called.")
        self.conversation_history.clear()
        self.uploaded_files.clear()
        self.fetched_urls.clear()
        self.fetched_image_urls.clear()

        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            if item:
                w = item.widget()
                if w:
                    w.deleteLater()
        self.source_widgets.clear()
        self.image_widgets.clear()
        self.image_offset = 0

        self.input_field.clear()
        self.init_search_bar.clear()
        self.init_uploaded_files_widget.clearFiles()
        self.chat_uploaded_files_widget.clearFiles()
        self.stack.setCurrentWidget(self.init_page)

        if self.settings_panel and self.settings_panel.isVisible():
            self.settings_panel.hide()
        if self.find_widget and self.find_widget.isVisible():
            self.find_widget.hide()
            self.clearHighlights()

    def toggleSettingsPanel(self):
        if self.settings_panel and self.settings_panel.isVisible():
            self.settings_panel.hide()
        else:
            if not self.settings_panel:
                self.createSettingsPanel()
            self.settings_panel.raise_()
            self.settings_panel.show()

    def createSettingsPanel(self):
        self.settings_panel = QFrame(self)
        self.settings_panel.setStyleSheet("QFrame { background-color: #2A2A2A; }")
        self.settings_panel.setFixedWidth(200)
        self.settings_panel.setGeometry(
            self.width() - 200,
            self.top_bar.height(),
            200,
            self.height() - self.top_bar.height()
        )
        self.settings_panel.setLayout(QVBoxLayout())
        self.settings_panel.layout().setContentsMargins(10, 10, 10, 10)
        self.settings_panel.layout().setAlignment(Qt.AlignTop)

        reload_button = QPushButton('Reload')
        reload_button.clicked.connect(self.reloadApp)
        self.settings_panel.layout().addWidget(reload_button)

        find_button = QPushButton('Find')
        find_button.clicked.connect(self.showFindDialog)
        self.settings_panel.layout().addWidget(find_button)

        self.settings_panel.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.settings_panel:
            self.settings_panel.setGeometry(
                self.width() - 200,
                self.top_bar.height(),
                200,
                self.height() - self.top_bar.height()
            )
        if self.find_widget:
            self.find_widget.move(10, self.top_bar.height() + 10)

    def showFindDialog(self):
        if self.find_widget and self.find_widget.isVisible():
            self.find_widget.hide()
            self.clearHighlights()
        else:
            if not hasattr(self, 'find_widget') or not self.find_widget:
                self.createFindWidget()
            self.find_widget.raise_()
            self.find_widget.show()
            self.find_widget.activateWindow()
            self.find_input.setFocus()

    def createFindWidget(self):
        self.find_widget = QFrame(self)
        self.find_widget.setStyleSheet("QFrame { background-color: #2E2E2E; border-radius: 10px; }")
        self.find_widget.setFixedHeight(50)
        self.find_widget.setGeometry(10, self.top_bar.height() + 10, 300, 50)

        layout = QHBoxLayout(self.find_widget)
        layout.setContentsMargins(10, 0, 10, 0)
        self.find_input = QLineEdit()
        self.find_input.setPlaceholderText('Find...')
        layout.addWidget(self.find_input)

        find_next_button = QPushButton('Next')
        find_next_button.clicked.connect(self.findNext)
        layout.addWidget(find_next_button)

        self.find_input.returnPressed.connect(self.findNext)
        self.search_results = []
        self.current_search_index = -1

    def findNext(self):
        txt = self.find_input.text()
        if not txt:
            return
        self.clearHighlights()
        self.search_results.clear()

        for i in range(self.scroll_layout.count()):
            w = self.scroll_layout.itemAt(i).widget()
            if isinstance(w, MessageWidget):
                matches = w.highlightText(txt)
                if matches:
                    self.search_results.append((w, matches))

        if self.search_results:
            self.current_search_index = (self.current_search_index + 1) % len(self.search_results)
            widget, _ = self.search_results[self.current_search_index]
            widget_position = widget.pos().y()
            self.scroll_area.verticalScrollBar().setValue(widget_position)
        else:
            self.current_search_index = -1

    def clearHighlights(self):
        for i in range(self.scroll_layout.count()):
            w = self.scroll_layout.itemAt(i).widget()
            if isinstance(w, MessageWidget):
                w.clearHighlights()

    def uploadFiles(self):
        print("[DEBUG] uploadFiles called.")
        if self.selected_model not in ['gpt-4o', 'gpt-4o-mini']:
            file_filter = (
                "Text/Code Files (*.txt *.py *.js);;"
                "All Files (*.txt *.py *.js)"
            )
        else:
            file_filter = (
                "All Supported Files (*.png *.jpg *.jpeg *.bmp *.gif *.txt *.py *.js *.pdf);;"
                "Images (*.png *.jpg *.jpeg *.bmp *.gif);;"
                "Text Files (*.txt);;"
                "Code Files (*.py *.js);;"
                "PDF Files (*.pdf)"
            )

        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Files to Upload",
            "",
            file_filter,
            options=options
        )
        if not files:
            return

        for file_path in files:
            ext = os.path.splitext(file_path)[1].lower()
            file_name = os.path.basename(file_path)

            if self.selected_model not in ['gpt-4o', 'gpt-4o-mini']:
                if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.pdf']:
                    self.showError("Selected model does not support image/PDF uploads.")
                    continue

            if ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                try:
                    encoded = self.encode_image(file_path)
                    self.uploaded_files.append({'type': 'image', 'data': encoded, 'name': file_name})
                    self.addFileToUI('image', file_name)
                except Exception as e:
                    self.showError(f"Failed to process image {file_name}: {str(e)}")

            elif ext in ['.txt', '.py', '.js']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    ftype = 'text' if ext == '.txt' else 'code'
                    self.uploaded_files.append({'type': ftype, 'data': content, 'name': file_name})
                    self.addFileToUI(ftype, file_name)
                except Exception as e:
                    self.showError(f"Failed to read file {file_name}: {str(e)}")

            elif ext == '.pdf':
                if self.selected_model not in ['gpt-4o', 'gpt-4o-mini']:
                    self.showError("Selected model does not support image/PDF uploads.")
                    continue
                try:
                    imgs = convert_from_path(file_path)
                    for idx, im in enumerate(imgs):
                        tmp_path = f"{file_path}_page_{idx+1}.png"
                        im.save(tmp_path, 'PNG')
                        encoded = self.encode_image(tmp_path)
                        self.uploaded_files.append({
                            'type': 'image',
                            'data': encoded,
                            'name': f"{file_name}_page_{idx+1}.png"
                        })
                        self.addFileToUI('image', f"{file_name}_page_{idx+1}.png")
                        os.remove(tmp_path)
                except Exception as e:
                    self.showError(f"Failed to process PDF {file_name}: {str(e)}")
            else:
                self.showError(f"Unsupported file type: {ext}")

    def addFileToUI(self, file_type, file_name):
        if self.stack.currentWidget() == self.init_page:
            self.uploaded_files_widgets['init'].addFile(file_type, file_name)
        else:
            self.uploaded_files_widgets['chat'].addFile(file_type, file_name)

    def encode_image(self, path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def closeEvent(self, event):
        print("[DEBUG] closeEvent called.")
        event.accept()

###############################################################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Alvely')

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.closeTab)

        new_tab_btn = QToolButton()
        new_tab_btn.setText('+')
        new_tab_btn.clicked.connect(self.addTab)
        self.tabs.setCornerWidget(new_tab_btn, Qt.TopRightCorner)

        self.setCentralWidget(self.tabs)
        self.addTab()

    def addTab(self):
        tab = ChatApp()
        index = self.tabs.addTab(tab, f'Tab {self.tabs.count() + 1}')
        self.tabs.setCurrentIndex(index)

    def closeTab(self, index):
        widget = self.tabs.widget(index)
        self.tabs.removeTab(index)
        if widget:
            widget.deleteLater()
        if self.tabs.count() == 0:
            self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
