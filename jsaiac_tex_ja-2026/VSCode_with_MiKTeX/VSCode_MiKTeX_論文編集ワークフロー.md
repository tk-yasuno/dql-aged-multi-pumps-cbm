# VSCodeとMiKTeXを使った学術論文編集ワークフロー

## 概要
VSCode上でMiKTeXを活用して学術論文を編集・コンパイルする統合開発環境の構築と運用手順をまとめました。従来のTeXエディタと比較して、モダンな開発環境の利便性を論文執筆に活用できることが確認できました。

## 🎯 このワークフローの利点

### 従来のTeX編集との比較
- **統合開発環境**: コード編集、プレビュー、ファイル管理が一体化
- **リアルタイムプレビュー**: 変更を即座に確認可能
- **Git連携**: バージョン管理が自然に統合
- **拡張機能**: シンタックスハイライト、自動補完、エラーチェック
- **プロジェクト管理**: 論文関連ファイルを統合的に管理

## 🛠️ 環境構成

### 必要コンポーネント
1. **Visual Studio Code**
   - LaTeX Workshop拡張機能
   - その他TeX関連拡張機能

2. **MiKTeX Distribution**
   - Windows用LaTeX配布版
   - パッケージ自動インストール機能

3. **プロジェクト構造**
```
jsaiac_tex_ja-2026/
├── pump_cbm_paper_2026v8_unify.tex    # メイン論文ファイル
├── jsaiac.sty                          # 学会テンプレート
├── jsai.bst                           # 参考文献スタイル
├── Figure_files/                       # 図表ディレクトリ
├── Experimental_results_3k/            # 実験結果
├── VSCode_with_MiKTeX/                # ワークフロー文書
│   ├── flow_VSCode_with_MiKTeX.jpg    # プロセスフロー図
│   └── VSCode_MiKTeX_論文編集ワークフロー.md
└── 260104_arXiv_upload/               # 投稿準備ディレクトリ
```

## 📋 詳細手順

### Step 1: 初期環境設定

#### 1.1 MiKTeX のインストールと設定
```powershell
# MiKTeX Console を使用してパッケージ管理
# 自動インストール設定: Ask me first → Yes (推奨)
```

#### 1.2 VSCode拡張機能の設定
- **LaTeX Workshop**: メインのTeX編集支援
- **LaTeX language support**: シンタックスハイライト
- **LaTeX Utilities**: 追加機能

#### 1.3 settings.json の設定例
```json
{
    "latex-workshop.latex.tools": [
        {
            "name": "platex",
            "command": "platex",
            "args": ["-synctex=1", "-interaction=nonstopmode", "-file-line-error", "%DOC%"]
        },
        {
            "name": "dvipdfmx",
            "command": "dvipdfmx",
            "args": ["%DOCFILE%"]
        }
    ],
    "latex-workshop.latex.recipes": [
        {
            "name": "platex -> dvipdfmx",
            "tools": ["platex", "dvipdfmx"]
        }
    ]
}
```

### Step 2: 論文プロジェクトの作成

#### 2.1 テンプレートファイルの準備
```tex
\documentclass[twocolumn]{article}
\usepackage{jsaiac}                    % 学会テンプレート
\usepackage[dvipdfmx]{graphicx}       % 図表挿入
\usepackage{url}                       % URL処理
\usepackage{amsmath}                   % 数式処理
\usepackage{amssymb}                   % 数学記号
\usepackage{color}                     % カラー対応
\usepackage[utf8]{inputenc}           % UTF-8入力
\usepackage[T1]{fontenc}              % フォントエンコーディング
```

#### 2.2 プロジェクト構造の整理
- **Figure_files/**: 図表の管理
- **Experimental_results_3k/**: 実験結果データ
- **260104_arXiv_upload/**: 投稿準備ファイル

### Step 3: 編集とコンパイルのワークフロー

#### 3.1 リアルタイム編集
1. `.tex`ファイルをVSCodeで開く
2. LaTeX Workshop拡張機能が自動的に有効化
3. `Ctrl+S`で保存と同時に自動コンパイル
4. プレビューパネルで即座に結果確認

#### 3.2 エラー処理
```
# コンパイルエラーが発生した場合
1. PROBLEMS パネルでエラー箇所を確認
2. LaTeX Workshop のログを参照
3. MiKTeX Console でパッケージ不足を解決
```

#### 3.3 参考文献の管理
```tex
\bibliographystyle{jsai}     % JSAIスタイル
\bibliography{references}    % .bibファイル参照
```

### Step 4: 投稿準備ワークフロー

#### 4.1 最終チェック項目
- [ ] すべての図表が正しく表示される
- [ ] 参考文献のフォーマットが適切
- [ ] ページ数が学会規定内
- [ ] コンパイルエラーが0件

#### 4.2 アーカイブ準備
```
260104_arXiv_upload/
├── pump_cbm_paper_2026v8_unify.pdf   # 最終PDF
├── pump_cbm_paper_2026v8_unify.tex   # TeXソース
├── jsaiac.sty                         # スタイルファイル
├── jsai.bst                          # 参考文献スタイル
└── Figure_files/                      # 全図表
```

## 🚀 高度な活用テクニック

### Git連携による版数管理
```bash
git add *.tex *.pdf
git commit -m "論文 v8 統合版完成"
git tag v8_unified
```

### ショートカットキーの活用
- `Ctrl+Alt+B`: コンパイル実行
- `Ctrl+Alt+V`: PDFプレビュー
- `Ctrl+Alt+C`: ログクリア

### 協同編集での注意点
```tex
% \usepackage{color} を使用した変更履歴
\textcolor{red}{追加内容}
\textcolor{blue}{修正内容}
```

## 📊 パフォーマンス比較

### 従来環境との作業効率比較
| 項目 | 従来のTeX環境 | VSCode+MiKTeX | 改善度 |
|------|-------------|---------------|--------|
| 編集～プレビュー時間 | 30-60秒 | 5-10秒 | **6倍向上** |
| エラー特定時間 | 2-5分 | 10-30秒 | **4-10倍向上** |
| ファイル管理効率 | 個別管理 | 統合管理 | **大幅改善** |
| 図表挿入作業 | 手動パス管理 | 自動補完 | **作業負荷軽減** |

## 🎓 学んだ教訓と今後の展開

### 重要な発見
1. **開発者ツールの学術応用**: プログラミング用IDEが論文執筆にも非常に有効
2. **統合環境の価値**: ファイル管理、編集、プレビューの一体化による効率向上
3. **自動化の重要性**: コンパイル、エラーチェックの自動化による集中力向上

### 今後の改善点
- [ ] 自動校正機能の追加検討
- [ ] 図表自動生成スクリプトとの連携
- [ ] 査読コメント管理システムの統合
- [ ] 複数言語対応（日本語・英語切り替え）

### 応用可能性
- **技術レポート作成**: 企業内技術文書
- **学位論文執筆**: 修士・博士論文への適用
- **国際会議投稿**: 英文論文への展開
- **書籍執筆**: より大規模な文書プロジェクト

## 📝 まとめ

VSCodeとMiKTeXを組み合わせた学術論文編集環境は、従来のTeX編集方法を大幅に改善する可能性を示しました。特に：

1. **作業効率の向上**: リアルタイムプレビューとエラー検出
2. **プロジェクト管理の統合化**: 論文関連ファイルの一元管理
3. **開発ツールの学術応用**: プログラマー向けツールの論文執筆への転用

この経験を通じて、技術分野における既存ツールの新たな活用可能性を発見することができました。今後も継続的に改善を重ね、より効率的な学術執筆環境の構築を目指します。

---

*作成日: 2026年1月4日*  
*対象論文: "Deep Reinforcement Learning Approach for Condition-Based Maintenance of Multi-Pump Equipment"*  
*環境: Windows 10, VSCode + MiKTeX*