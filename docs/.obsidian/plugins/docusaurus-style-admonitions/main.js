/*
THIS IS A GENERATED/BUNDLED FILE BY ESBUILD
if you want to view the source, please visit the github repository of this plugin
*/

var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// main.ts
var main_exports = {};
__export(main_exports, {
  default: () => DocusaurusAdmonitionsPlugin
});
module.exports = __toCommonJS(main_exports);
var import_obsidian = require("obsidian");
var import_view = require("@codemirror/view");
var import_view2 = require("@codemirror/view");
var DEFAULT_SETTINGS = {
  enabledAdmonitions: {
    note: true,
    tip: true,
    info: true,
    warning: true,
    danger: true
  },
  customCSS: true,
  enableCodeBlockSyntax: false
  // Standardmäßig deaktiviert
};
var DocusaurusAdmonitionsPlugin = class extends import_obsidian.Plugin {
  /** Wird aufgerufen, wenn das Plugin geladen wird. */
  async onload() {
    console.log("Docusaurus Admonitions Plugin geladen");
    await this.loadSettings();
    this.injectStyles();
    if (this.settings.enableCodeBlockSyntax) {
      this.registerMarkdownCodeBlockProcessor("note", this.processAdmonition.bind(this, "note"));
      this.registerMarkdownCodeBlockProcessor("tip", this.processAdmonition.bind(this, "tip"));
      this.registerMarkdownCodeBlockProcessor("info", this.processAdmonition.bind(this, "info"));
      this.registerMarkdownCodeBlockProcessor("warning", this.processAdmonition.bind(this, "warning"));
      this.registerMarkdownCodeBlockProcessor("danger", this.processAdmonition.bind(this, "danger"));
    }
    this.registerLivePreviewRenderer();
    this.addSettingTab(new DocusaurusAdmonitionsSettingTab(this.app, this));
    window.inspectDocusaurusAdmonitions = this.inspectAdmonitions;
    console.log("Debug-Funktion verf\xFCgbar: window.inspectDocusaurusAdmonitions()");
    setTimeout(() => {
      console.log(this.inspectAdmonitions());
    }, 3e3);
  }
  /** Erstellt & injiziert CSS-Styles für Reading Mode und Live Preview. */
  injectStyles() {
    const readingModeStyle = document.createElement("style");
    readingModeStyle.id = "docusaurus-admonitions-styles";
    readingModeStyle.textContent = `
    .docusaurus-admonition {
        margin-bottom: 1em;
        padding: 16px;
        border-radius: 8px;
        border-left: 0;
        background-color: var(--background-secondary);
        position: relative;
        overflow: hidden;
    }

    /* Farbige Seitenleiste f\xFCr jeden Typ */
    .docusaurus-admonition::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
    }

    .docusaurus-admonition-note::before { background-color: #3578e5; }
    .docusaurus-admonition-tip::before { background-color: #00a400; }
    .docusaurus-admonition-info::before { background-color: #3578e5; }
    .docusaurus-admonition-warning::before { background-color: #e6a700; }
    .docusaurus-admonition-danger::before { background-color: #fa383e; }

    /* Titel mit Icons und Farben */
    .docusaurus-admonition-title {
        margin-top: 0 !important;
        margin-bottom: 14px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        font-size: 0.8em !important;
        line-height: 1.5 !important;
        display: flex !important;
        align-items: center !important;
    }

    /* Icons f\xFCr jeden Admonition-Typ */
    .docusaurus-admonition-title::before {
        content: '' !important;
        margin-right: 8px !important;
        width: 20px !important;
        height: 20px !important;
        min-width: 20px !important;
        display: inline-block !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
        background-size: contain !important;
    }

    /* Icon f\xFCr NOTE (Info Circle) */
    .docusaurus-admonition-note .docusaurus-admonition-title {
        color: #3578e5;
    }
    .docusaurus-admonition-note .docusaurus-admonition-title::before {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 512 512'%3E%3Cpath fill='%233578e5' d='M256 0C114.6 0 0 114.6 0 256s114.6 256 256 256s256-114.6 256-256S397.4 0 256 0zM256 128c17.67 0 32 14.33 32 32c0 17.67-14.33 32-32 32S224 177.7 224 160C224 142.3 238.3 128 256 128zM296 384h-80C202.8 384 192 373.3 192 360s10.75-24 24-24h16v-64H224c-13.25 0-24-10.75-24-24S210.8 224 224 224h32c13.25 0 24 10.75 24 24v88h16c13.25 0 24 10.75 24 24S309.3 384 296 384z'%3E%3C/path%3E%3C/svg%3E");
    }

    /* Icon f\xFCr TIP (Lightbulb) */
    .docusaurus-admonition-tip .docusaurus-admonition-title {
        color: #00a400;
    }
    .docusaurus-admonition-tip .docusaurus-admonition-title::before {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 384 512'%3E%3Cpath fill='%2300a400' d='M112.1 454.3c0 6.297 1.816 12.44 5.284 17.69l17.14 25.69c5.25 7.875 17.17 14.28 26.64 14.28h61.67c9.438 0 21.36-6.401 26.61-14.28l17.08-25.68c2.938-4.438 5.348-12.37 5.348-17.7L272 415.1h-160L112.1 454.3zM191.4 .0132C89.44 .3257 16 82.97 16 175.1c0 44.38 16.44 84.84 43.56 115.8c16.53 18.84 42.34 58.23 52.22 91.45c.0313 .25 .0938 .5166 .125 .7823h160.2c.0313-.2656 .0938-.5166 .125-.7823c9.875-33.22 35.69-72.61 52.22-91.45C351.6 260.8 368 220.4 368 175.1C368 78.61 288.9 .0132 191.4 .0132zM192 96.01c-44.13 0-80 35.89-80 79.1C112 184.8 104.8 192 96 192S80 184.8 80 176c0-61.76 50.25-111.1 112-111.1c8.844 0 16 7.159 16 16S200.8 96.01 192 96.01z'/%3E%3C/svg%3E");
		}

		/* Icon f\xFCr INFO (Info Circle) - gleich wie NOTE */
		.docusaurus-admonition-info .docusaurus-admonition-title {
			color: #3578e5;
			}
			.docusaurus-admonition-info .docusaurus-admonition-title::before {
background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 512 512'%3E%3Cpath fill='%233578e5' d='M256 0C114.6 0 0 114.6 0 256s114.6 256 256 256s256-114.6 256-256S397.4 0 256 0zM256 128c17.67 0 32 14.33 32 32c0 17.67-14.33 32-32 32S224 177.7 224 160C224 142.3 238.3 128 256 128zM296 384h-80C202.8 384 192 373.3 192 360s10.75-24 24-24h16v-64H224c-13.25 0-24-10.75-24-24S210.8 224 224 224h32c13.25 0 24 10.75 24 24v88h16c13.25 0 24 10.75 24 24S309.3 384 296 384z'%3E%3C/path%3E%3C/svg%3E");    }

    /* Icon f\xFCr WARNING (Exclamation Triangle) */
    .docusaurus-admonition-warning .docusaurus-admonition-title {
        color: #e6a700;
    }
    .docusaurus-admonition-warning .docusaurus-admonition-title::before {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512'%3E%3Cpath fill='%23e6a700' d='M506.3 417l-213.3-364c-16.33-28-57.54-28-73.98 0l-213.2 364C-10.59 444.9 9.849 480 42.74 480h426.6C502.1 480 522.6 445 506.3 417zM232 168c0-13.25 10.75-24 24-24S280 154.8 280 168v128c0 13.25-10.75 24-24 24S232 309.3 232 296V168zM256 416c-17.36 0-31.44-14.08-31.44-31.44c0-17.36 14.07-31.44 31.44-31.44s31.44 14.08 31.44 31.44C287.4 401.9 273.4 416 256 416z'/%3E%3C/svg%3E");
    }

    /* Icon f\xFCr DANGER (Exclamation Circle) */
    .docusaurus-admonition-danger .docusaurus-admonition-title {
        color: #fa383e;
    }
    .docusaurus-admonition-danger .docusaurus-admonition-title::before {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512'%3E%3Cpath fill='%23fa383e' d='M256 0C114.6 0 0 114.6 0 256s114.6 256 256 256s256-114.6 256-256S397.4 0 256 0zM232 152C232 138.8 242.8 128 256 128s24 10.75 24 24v128c0 13.25-10.75 24-24 24S232 293.3 232 280V152zM256 400c-17.36 0-31.44-14.08-31.44-31.44c0-17.36 14.07-31.44 31.44-31.44s31.44 14.08 31.44 31.44C287.4 385.9 273.4 400 256 400z'/%3E%3C/svg%3E");
    }

    /* Inhalt-Styling */
    .docusaurus-admonition-content p:last-child {
        margin-bottom: 0;
    }
`;
    document.head.appendChild(readingModeStyle);
    const livePreviewStyle = document.createElement("style");
    livePreviewStyle.id = "docusaurus-admonitions-editor-styles";
    livePreviewStyle.textContent = `
    /* =============== Allgemeine Styles f\xFCr Admonition-Zeilen =============== */
    .admonition-note-start, .admonition-note-end, .admonition-note-content,
    .admonition-tip-start, .admonition-tip-end, .admonition-tip-content,
    .admonition-info-start, .admonition-info-end, .admonition-info-content,
    .admonition-warning-start, .admonition-warning-end, .admonition-warning-content,
    .admonition-danger-start, .admonition-danger-end, .admonition-danger-content {
        padding-left: 20px !important; /* Erh\xF6hter Abstand f\xFCr alle Admonition-Zeilen */
        position: relative;
    }

    /* =============== Linien an der Seite f\xFCr jeden Typ =============== */
    .admonition-note-start::before, .admonition-note-end::before, .admonition-note-content::before,
    .admonition-tip-start::before, .admonition-tip-end::before, .admonition-tip-content::before,
    .admonition-info-start::before, .admonition-info-end::before, .admonition-info-content::before,
    .admonition-warning-start::before, .admonition-warning-end::before, .admonition-warning-content::before,
    .admonition-danger-start::before, .admonition-danger-end::before, .admonition-danger-content::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 4px;
    }

    /* Note Style mit Icon */
    .admonition-note-start, .admonition-note-end, .admonition-note-content {
        background-color: rgba(53, 120, 229, 0.05);
    }
    .admonition-note-start::before, .admonition-note-end::before, .admonition-note-content::before {
        background-color: #3578e5;
    }
	.admonition-note-start {
		font-weight: bold;
		color: #3578e5 !important;
		padding-left: 44px !important; /* Mehr Platz f\xFCr das Icon */
		position: relative; /* Notwendig f\xFCr die absolute Positionierung */
		background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512' width='16' height='16'%3E%3Cpath fill='%233578e5' d='M256 0C114.6 0 0 114.6 0 256s114.6 256 256 256s256-114.6 256-256S397.4 0 256 0zM256 128c17.67 0 32 14.33 32 32c0 17.67-14.33 32-32 32S224 177.7 224 160C224 142.3 238.3 128 256 128zM296 384h-80C202.8 384 192 373.3 192 360s10.75-24 24-24h16v-64H224c-13.25 0-24-10.75-24-24S210.8 224 224 224h32c13.25 0 24 10.75 24 24v88h16c13.25 0 24 10.75 24 24S309.3 384 296 384z'/%3E%3C/svg%3E");
		background-repeat: no-repeat;
		background-position: 20px center !important; /* Icon-Position anpassen */
		background-size: 16px;
	}

	/* Entferne das alte ::after */
	.admonition-note-start::after {
		content: none;
	}

    /* Tip Style mit Icon */
    .admonition-tip-start, .admonition-tip-end, .admonition-tip-content {
        background-color: rgba(0, 164, 0, 0.05);
    }
    .admonition-tip-start::before, .admonition-tip-end::before, .admonition-tip-content::before {
        background-color: #00a400;
    }
	.admonition-tip-start {
		font-weight: bold;
		color: #00a400 !important;
		padding-left: 44px !important; /* Mehr Platz f\xFCr das Icon */
		position: relative; /* Notwendig f\xFCr die absolute Positionierung */
		background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 384 512' width='16' height='16'%3E%3Cpath fill='%2300a400' d='M112.1 454.3c0 6.297 1.816 12.44 5.284 17.69l17.14 25.69c5.25 7.875 17.17 14.28 26.64 14.28h61.67c9.438 0 21.36-6.401 26.61-14.28l17.08-25.68c2.938-4.438 5.348-12.37 5.348-17.7L272 415.1h-160L112.1 454.3zM191.4 .0132C89.44 .3257 16 82.97 16 175.1c0 44.38 16.44 84.84 43.56 115.8c16.53 18.84 42.34 58.23 52.22 91.45c.0313 .25 .0938 .5166 .125 .7823h160.2c.0313-.2656 .0938-.5166 .125-.7823c9.875-33.22 35.69-72.61 52.22-91.45C351.6 260.8 368 220.4 368 175.1C368 78.61 288.9 .0132 191.4 .0132zM192 96.01c-44.13 0-80 35.89-80 79.1C112 184.8 104.8 192 96 192S80 184.8 80 176c0-61.76 50.25-111.1 112-111.1c8.844 0 16 7.159 16 16S200.8 96.01 192 96.01z'/%3E%3C/svg%3E");
		background-repeat: no-repeat;
		background-position: 20px center !important; /* Icon-Position anpassen */
		background-size: 16px;
	}

	/* Entferne das alte ::after */
	.admonition-tip-start::after {
		content: none;
	}

    /* Info Style mit Icon */
    .admonition-info-start, .admonition-info-end, .admonition-info-content {
        background-color: rgba(53, 120, 229, 0.05);
    }
    .admonition-info-start::before, .admonition-info-end::before, .admonition-info-content::before {
        background-color: #3578e5;
    }
	.admonition-info-start {
		font-weight: bold;
		color: #3578e5 !important;
		padding-left: 44px !important; /* Mehr Platz f\xFCr das Icon */
		position: relative; /* Notwendig f\xFCr die absolute Positionierung */
		background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512' width='16' height='16'%3E%3Cpath fill='%233578e5' d='M256 0C114.6 0 0 114.6 0 256s114.6 256 256 256s256-114.6 256-256S397.4 0 256 0zM256 128c17.67 0 32 14.33 32 32c0 17.67-14.33 32-32 32S224 177.7 224 160C224 142.3 238.3 128 256 128zM296 384h-80C202.8 384 192 373.3 192 360s10.75-24 24-24h16v-64H224c-13.25 0-24-10.75-24-24S210.8 224 224 224h32c13.25 0 24 10.75 24 24v88h16c13.25 0 24 10.75 24 24S309.3 384 296 384z'/%3E%3C/svg%3E");
		background-repeat: no-repeat;
		background-position: 20px center !important; /* Icon-Position anpassen */
		background-size: 16px;
	}

	/* Entferne das alte ::after */
	.admonition-info-start::after {
		content: none;
	}

    /* Warning Style mit Icon */
    .admonition-warning-start, .admonition-warning-end, .admonition-warning-content {
        background-color: rgba(230, 167, 0, 0.05);
    }
    .admonition-warning-start::before, .admonition-warning-end::before, .admonition-warning-content::before {
        background-color: #e6a700;
    }
	.admonition-warning-start {
		font-weight: bold;
		color: #e6a700 !important;
		padding-left: 44px !important; /* Mehr Platz f\xFCr das Icon */
		position: relative; /* Notwendig f\xFCr die absolute Positionierung */
		background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512' width='16' height='16'%3E%3Cpath fill='%23e6a700' d='M506.3 417l-213.3-364c-16.33-28-57.54-28-73.98 0l-213.2 364C-10.59 444.9 9.849 480 42.74 480h426.6C502.1 480 522.6 445 506.3 417zM232 168c0-13.25 10.75-24 24-24S280 154.8 280 168v128c0 13.25-10.75 24-24 24S232 309.3 232 296V168zM256 416c-17.36 0-31.44-14.08-31.44-31.44c0-17.36 14.07-31.44 31.44-31.44s31.44 14.08 31.44 31.44C287.4 401.9 273.4 416 256 416z'/%3E%3C/svg%3E");
		background-repeat: no-repeat;
		background-position: 20px center !important; /* Icon-Position anpassen */
		background-size: 16px;
	}

	/* Entferne das alte ::after */
	.admonition-warning-start::after {
		content: none;
	}

    /* Danger Style mit Icon */
    .admonition-danger-start, .admonition-danger-end, .admonition-danger-content {
        background-color: rgba(250, 56, 62, 0.05);
    }
    .admonition-danger-start::before, .admonition-danger-end::before, .admonition-danger-content::before {
        background-color: #fa383e;
    }
	.admonition-danger-start {
		font-weight: bold;
		color: #fa383e !important;
		padding-left: 44px !important; /* Mehr Platz f\xFCr das Icon */
		position: relative; /* Notwendig f\xFCr die absolute Positionierung */
		background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512' width='16' height='16'%3E%3Cpath fill='%23fa383e' d='M256 0C114.6 0 0 114.6 0 256s114.6 256 256 256s256-114.6 256-256S397.4 0 256 0zM232 152C232 138.8 242.8 128 256 128s24 10.75 24 24v128c0 13.25-10.75 24-24 24S232 293.3 232 280V152zM256 400c-17.36 0-31.44-14.08-31.44-31.44c0-17.36 14.07-31.44 31.44-31.44s31.44 14.08 31.44 31.44C287.4 385.9 273.4 400 256 400z'/%3E%3C/svg%3E");
		background-repeat: no-repeat;
		background-position: 20px center !important; /* Icon-Position anpassen */
		background-size: 16px;
	}

	/* Entferne das alte ::after */
	.admonition-danger-start::after {
		content: none;
	}`;
    document.head.appendChild(livePreviewStyle);
    console.log("Docusaurus Admonitions: Styles injected.");
  }
  /** Verarbeitet Code-Blöcke (z. B. ```note ... ```), um Reading Mode-Admonitions zu erzeugen. */
  async processAdmonition(type, source, el, ctx) {
    var _a;
    if (!this.settings.enabledAdmonitions[type]) {
      return;
    }
    el.empty();
    const admonitionDiv = el.createDiv({
      cls: ["docusaurus-admonition", `docusaurus-admonition-${type}`]
    });
    const titleDiv = admonitionDiv.createDiv({
      cls: "docusaurus-admonition-title"
    });
    titleDiv.textContent = type.toUpperCase();
    console.log(`Admonition erstellt: ${type}`, {
      "Hat Titel?": admonitionDiv.querySelector(".docusaurus-admonition-title") !== null,
      "Elternelement": (_a = el.parentElement) == null ? void 0 : _a.tagName,
      "CSS geladen?": document.getElementById("docusaurus-admonitions-styles") !== null
    });
    const contentDiv = admonitionDiv.createDiv({
      cls: "docusaurus-admonition-content"
    });
    await import_obsidian.MarkdownRenderer.renderMarkdown(source, contentDiv, ctx.sourcePath, this);
  }
  /** Verarbeitet die :::type-Syntax in Reading Mode. */
  async processCustomAdmonitionSyntax(el, ctx) {
    var _a, _b;
    const paragraphs = el.querySelectorAll("p");
    for (let i = 0; i < paragraphs.length; i++) {
      const p = paragraphs[i];
      const text = (_a = p.textContent) == null ? void 0 : _a.trim();
      if (!text || !text.startsWith(":::"))
        continue;
      const match = text.match(/^:::(note|tip|info|warning|danger)(?:\s|$)/);
      if (!match)
        continue;
      const type = match[1];
      const singleLineMatch = text.match(/^:::(note|tip|info|warning|danger)\s+([\s\S]+?)\s+:::$/);
      if (singleLineMatch) {
        const singleType = singleLineMatch[1];
        const content2 = singleLineMatch[2];
        if (!this.settings.enabledAdmonitions[singleType]) {
          continue;
        }
        const admonitionDiv2 = el.createDiv({
          cls: ["docusaurus-admonition", `docusaurus-admonition-${singleType}`]
        });
        admonitionDiv2.createDiv({
          cls: "docusaurus-admonition-title",
          text: singleType.toUpperCase()
        });
        const contentDiv2 = admonitionDiv2.createDiv({ cls: "docusaurus-admonition-content" });
        await import_obsidian.MarkdownRenderer.renderMarkdown(content2, contentDiv2, ctx.sourcePath, this);
        p.replaceWith(admonitionDiv2);
        continue;
      }
      let endIndex = -1;
      let content = [];
      for (let j = i + 1; j < paragraphs.length; j++) {
        const endText = (_b = paragraphs[j].textContent) == null ? void 0 : _b.trim();
        if (endText === ":::") {
          endIndex = j;
          break;
        } else {
          content.push(paragraphs[j]);
        }
      }
      if (endIndex === -1)
        continue;
      if (!this.settings.enabledAdmonitions[type]) {
        continue;
      }
      const admonitionDiv = el.createDiv({
        cls: ["docusaurus-admonition", `docusaurus-admonition-${type}`]
      });
      admonitionDiv.createDiv({
        cls: "docusaurus-admonition-title",
        text: type.toUpperCase()
      });
      const contentDiv = admonitionDiv.createDiv({ cls: "docusaurus-admonition-content" });
      for (let k = 0; k < content.length; k++) {
        contentDiv.appendChild(content[k].cloneNode(true));
      }
      p.replaceWith(admonitionDiv);
      content.forEach((el2) => el2.remove());
      paragraphs[endIndex].remove();
      i = endIndex;
    }
  }
  /** Registriert Post-Processor & CodeMirror-Dekorationen für Live Preview. */
  registerLivePreviewRenderer() {
    this.registerMarkdownPostProcessor((el, ctx) => {
      this.processCustomAdmonitionSyntax(el, ctx);
    });
    try {
      const pluginExtension = createAdmonitionViewPlugin();
      this.registerEditorExtension([pluginExtension]);
      console.log("Docusaurus Admonitions: Live Preview aktiviert.");
    } catch (e) {
      console.error("Docusaurus Admonitions: Live Preview konnte nicht aktiviert werden:", e);
      this.addCSS_Fallback();
    }
  }
  /** Fallback: Einfaches CSS, falls das ViewPlugin scheitert */
  addCSS_Fallback() {
    const fallbackStyles = document.createElement("style");
    fallbackStyles.id = "docusaurus-admonitions-fallback-styles";
    fallbackStyles.textContent = `
			/* Minimaler Fallback: Hebt nur Zeilen mit :::note etc. farbig hervor. */
			.cm-line:has(.cm-string:contains(':::note')) {
				color: #448aff;
				border-left: 3px solid #448aff;
				font-weight: bold;
			}
			/* Weitere Admonition-Typen analog... */
		`;
    document.head.appendChild(fallbackStyles);
    console.log("Docusaurus Admonitions: Fallback-CSS hinzugef\xFCgt.");
  }
  /** Beim Deaktivieren: CSS & Debug-Elemente entfernen. */
  onunload() {
    const styleIds = [
      "docusaurus-admonitions-styles",
      "docusaurus-admonitions-editor-styles",
      "docusaurus-admonitions-fallback-styles"
    ];
    styleIds.forEach((id) => {
      const el = document.getElementById(id);
      if (el)
        el.remove();
    });
    console.log("Docusaurus Admonitions: Plugin entladen.");
  }
  /** Einstellungen laden */
  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }
  /** Einstellungen speichern */
  async saveSettings() {
    await this.saveData(this.settings);
  }
  /** Debugging-Funktion für dein Dokument */
  testDocumentStructure() {
    const debugBtn = document.createElement("button");
    debugBtn.textContent = "DEBUG";
    debugBtn.style.position = "fixed";
    debugBtn.style.top = "10px";
    debugBtn.style.right = "10px";
    debugBtn.style.zIndex = "1000";
    debugBtn.addEventListener("click", () => {
      const activeLeaf = this.app.workspace.activeLeaf;
      if (!activeLeaf)
        return;
      const view = activeLeaf.view;
      if (view.getViewType() === "markdown") {
        console.log("Aktuelle Dokument-Struktur:");
        const previewEl = view.containerEl.querySelector(".markdown-preview-view");
        if (previewEl) {
          console.log(previewEl.innerHTML);
          const paragraphs = previewEl.querySelectorAll("p");
          paragraphs.forEach((p, i) => {
            console.log(`P[${i}]: "${p.textContent}"`);
          });
        }
      }
    });
    document.body.appendChild(debugBtn);
  }
  inspectAdmonitions() {
    const admonitions = document.querySelectorAll(".docusaurus-admonition");
    console.log(`${admonitions.length} Admonitions gefunden`);
    admonitions.forEach((adm, i) => {
      var _a;
      const type = ((_a = Array.from(adm.classList).find((c) => c.startsWith("docusaurus-admonition-") && c !== "docusaurus-admonition")) == null ? void 0 : _a.replace("docusaurus-admonition-", "")) || "unbekannt";
      console.log(`Admonition #${i} (${type}):`);
      console.log("- HTML:", adm.outerHTML);
      console.log("- Titel vorhanden:", adm.querySelector(".docusaurus-admonition-title") !== null);
      console.log(
        "- Computed Style f\xFCr Titel:",
        window.getComputedStyle(
          adm.querySelector(".docusaurus-admonition-title") || adm
        )
      );
    });
    return `${admonitions.length} Admonitions inspiziert`;
  }
};
var DocusaurusAdmonitionsSettingTab = class extends import_obsidian.PluginSettingTab {
  constructor(app, plugin) {
    super(app, plugin);
    this.plugin = plugin;
  }
  display() {
    const { containerEl } = this;
    containerEl.empty();
    containerEl.createEl("h2", { text: "Docusaurus Admonitions Einstellungen" });
    const desc = "Aktiviert die :::SYNTAX Admonition";
    const types = ["note", "tip", "info", "warning", "danger"];
    types.forEach((type) => {
      new import_obsidian.Setting(containerEl).setName(`${type.toUpperCase()} Admonition`).setDesc(`${desc.replace("SYNTAX", type)}`).addToggle(
        (toggle) => toggle.setValue(this.plugin.settings.enabledAdmonitions[type]).onChange(async (value) => {
          this.plugin.settings.enabledAdmonitions[type] = value;
          await this.plugin.saveSettings();
        })
      );
    });
    containerEl.createEl("h3", { text: "Syntax-Optionen" });
    new import_obsidian.Setting(containerEl).setName("Code-Block-Syntax aktivieren").setDesc("Erm\xF6glicht die Verwendung von ```note Code-Bl\xF6cken f\xFCr Admonitions. Diese Syntax ist nicht Docusaurus-kompatibel.").addToggle(
      (toggle) => toggle.setValue(this.plugin.settings.enableCodeBlockSyntax).onChange(async (value) => {
        this.plugin.settings.enableCodeBlockSyntax = value;
        await this.plugin.saveSettings();
        new import_obsidian.Notice("Bitte Obsidian neu starten, um die Syntax-\xC4nderung wirksam zu machen.");
      })
    );
  }
};
function createAdmonitionViewPlugin() {
  return import_view2.ViewPlugin.fromClass(
    class {
      constructor(view) {
        this.decorations = computeAdmonitionDecorations(view);
      }
      update(update) {
        if (update.docChanged || update.viewportChanged) {
          this.decorations = computeAdmonitionDecorations(update.view);
        }
      }
    },
    {
      decorations: (v) => v.decorations
    }
  );
}
function computeAdmonitionDecorations(view) {
  const types = ["note", "tip", "info", "warning", "danger"];
  const decorations = [];
  const doc = view.state.doc;
  let pos = 0;
  while (pos < doc.length) {
    const line = doc.lineAt(pos);
    const text = line.text;
    let foundStart = false;
    for (const t of types) {
      const startRegex = new RegExp(`^:::${t}(?:\\s|$)`);
      if (startRegex.test(text)) {
        decorations.push(
          import_view.Decoration.line({
            attributes: { class: `admonition-${t}-start` }
          }).range(line.from)
        );
        foundStart = true;
        let innerPos = line.to + 1;
        while (innerPos < doc.length) {
          const innerLine = doc.lineAt(innerPos);
          const innerText = innerLine.text.trim();
          if (innerText === ":::") {
            decorations.push(
              import_view.Decoration.line({
                attributes: { class: `admonition-${t}-end` }
              }).range(innerLine.from)
            );
            break;
          } else {
            decorations.push(
              import_view.Decoration.line({
                attributes: { class: `admonition-${t}-content` }
              }).range(innerLine.from)
            );
          }
          innerPos = innerLine.to + 1;
        }
        break;
      }
    }
    pos = line.to + 1;
  }
  return import_view.Decoration.set(decorations, true);
}


/* nosourcemap */