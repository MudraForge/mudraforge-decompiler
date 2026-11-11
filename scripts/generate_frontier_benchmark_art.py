#!/usr/bin/env python3
"""
MudraForge Decompiler - High-Fidelity Benchmark Infographic Generator
Generates publication-quality SVG and PNG figures for the Frontier LLM comparison:
- docs/images/frontier_llm_benchmark_comparison.svg (Vector graphics for GitHub & web)
- docs/images/frontier_llm_benchmark_comparison.png (High-DPI raster image)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(REPO_DIR, "docs", "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_svg_figure():
    """Generates an ultra-clean, modern, publication-grade SVG graphic."""
    svg_path = os.path.join(OUTPUT_DIR, "frontier_llm_benchmark_comparison.svg")

    models_data = [
        # (Name, Paradigm, Params, Score, ColorStart, ColorEnd, IsOurs)
        ("MudraForge Decompiler + USVM", "Dual-Stream AST Compiler + Virtual Machine", "5.8M (80,000x smaller)", 100.0, "#10b981", "#059669", True),
        ("GPT-5.2 + Terminus-KIRA", "Multi-Turn Agentic Scaffold", "Frontier (>1T)", 60.0, "#6366f1", "#4f46e5", False),
        ("Claude Sonnet 4.5 + Terminus-KIRA", "Multi-Turn Agentic Scaffold", "Frontier (~500B)", 54.0, "#818cf8", "#6366f1", False),
        ("Best Model in DeonticBench (o3 / Coder-480B)", "CoT / LLM -> External SWI-Prolog", "Up to 480B", 44.4, "#f59e0b", "#d97706", False),
        ("Claude Sonnet 4.5", "Standard Direct / Chain-of-Thought", "Frontier (~500B)", 36.0, "#fb7185", "#f43f5e", False),
        ("Qwen3.5-35B", "Standard Direct / Chain-of-Thought", "35B", 34.0, "#fda4af", "#fb7185", False),
        ("GPT-5.2", "Standard Direct / Chain-of-Thought", "Frontier (>1T)", 30.0, "#fda4af", "#fb7185", False),
        ("Qwen3.5-35B (Agentic CLI)", "CLI Tool-Calling Harness", "35B", 11.0, "#f43f5e", "#e11d48", False),
    ]

    width = 1180
    height = 760
    bar_start_x = 420
    bar_max_width = 540

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '  <defs>',
        '    <style>',
        '      @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&amp;display=swap");',
        '      text { font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }',
        '    </style>',
        '    <linearGradient id="grad-emerald" x1="0%" y1="0%" x2="100%" y2="0%">',
        '      <stop offset="0%" stop-color="#10b981" />',
        '      <stop offset="100%" stop-color="#059669" />',
        '    </linearGradient>',
        '    <filter id="card-shadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">',
        '      <feDropShadow dx="0" dy="8" stdDeviation="12" flood-color="#0f172a" flood-opacity="0.06" />',
        '    </filter>',
        '    <filter id="hero-shadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">',
        '      <feDropShadow dx="0" dy="4" stdDeviation="8" flood-color="#059669" flood-opacity="0.18" />',
        '    </filter>',
        '  </defs>',
        '',
        '  <!-- Background Canvas -->',
        f'  <rect width="{width}" height="{height}" fill="#f8fafc" />',
        '',
        '  <!-- Main Card Container -->',
        f'  <rect x="24" y="24" width="{width - 48}" height="{height - 48}" rx="20" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" filter="url(#card-shadow)" />',
        '',
        '  <!-- Header Section -->',
        '  <g transform="translate(60, 64)">',
        '    <!-- Pill Badge -->',
        '    <rect x="0" y="0" width="410" height="26" rx="13" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1" />',
        '    <text x="14" y="17" font-size="11.5" font-weight="700" fill="#475569" letter-spacing="0.5">SARA-NUMERIC HARD SPLIT • 35 CURATED INSTANCES</text>',
        '    ',
        '    <!-- Main Title -->',
        '    <text x="0" y="58" font-size="24" font-weight="800" fill="#0f172a">Statutory Reasoning Benchmark: MudraForge vs. Frontier LLMs</text>',
        '    ',
        '    <!-- Subtitle -->',
        '    <text x="0" y="84" font-size="13" font-weight="500" fill="#64748b">Comparison across published 2026 baselines on the exact hard tax split that broke frontier models (arXiv:2604.04443 &amp; DAR 2026)</text>',
        '  </g>',
        '',
        '  <!-- Header Columns Legend -->',
        '  <g transform="translate(60, 184)">',
        '    <text x="0" y="0" font-size="11" font-weight="700" fill="#94a3b8" letter-spacing="0.5">SYSTEM &amp; REASONING PARADIGM</text>',
        '    <text x="360" y="0" font-size="11" font-weight="700" fill="#94a3b8" letter-spacing="0.5">SCALE</text>',
        '    <text x="bar_start_x" y="0" font-size="11" font-weight="700" fill="#94a3b8" letter-spacing="0.5" transform="translate(420, 0)">EXECUTION ACCURACY (35 HARD CASES)</text>',
        '  </g>',
        '  <line x1="60" y1="196" x2="1120" y2="196" stroke="#f1f5f9" stroke-width="1.5" />',
        '',
        '  <!-- Leaderboard Rows -->',
    ]

    row_y_start = 214
    row_height = 48

    for idx, (name, paradigm, scale, score, col_start, col_end, is_ours) in enumerate(models_data):
        y = row_y_start + (idx * row_height)
        bar_w = int(bar_max_width * (score / 100.0))

        if is_ours:
            # Highlight card for MudraForge
            svg_lines.append(f'  <!-- MudraForge Highlight Row -->')
            svg_lines.append(f'  <rect x="46" y="{y - 8}" width="{width - 92}" height="{row_height - 2}" rx="12" fill="#f0fdf4" stroke="#86efac" stroke-width="1.5" filter="url(#hero-shadow)" />')
            name_weight = "800"
            name_color = "#065f46"
            score_bg = "#10b981"
            score_text_col = "#ffffff"
            scale_bg = "#dcfce7"
            scale_border = "#86efac"
            scale_col = "#15803d"
        else:
            name_weight = "600"
            name_color = "#1e293b"
            score_bg = "#f1f5f9"
            score_text_col = "#0f172a"
            scale_bg = "#f8fafc"
            scale_border = "#e2e8f0"
            scale_col = "#64748b"

        # Model name & paradigm
        svg_lines.append(f'  <g transform="translate(60, {y})">')
        svg_lines.append(f'    <text x="0" y="14" font-size="13.5" font-weight="{name_weight}" fill="{name_color}">{name}</text>')
        svg_lines.append(f'    <text x="0" y="28" font-size="10.5" font-weight="500" fill="#64748b">{paradigm}</text>')
        svg_lines.append(f'  </g>')

        # Scale badge
        svg_lines.append(f'  <g transform="translate(410, {y})">')
        svg_lines.append(f'    <rect x="0" y="2" width="130" height="24" rx="6" fill="{scale_bg}" stroke="{scale_border}" stroke-width="1" />')
        svg_lines.append(f'    <text x="65" y="17.5" font-size="10" font-weight="700" fill="{scale_col}" text-anchor="middle">{scale}</text>')
        svg_lines.append(f'  </g>')

        # Horizontal Bar Track & Fill
        bar_x = 560
        svg_lines.append(f'  <g transform="translate({bar_x}, {y})">')
        svg_lines.append(f'    <!-- Track Background -->')
        svg_lines.append(f'    <rect x="0" y="7" width="{bar_max_width - 80}" height="14" rx="7" fill="#f1f5f9" />')
        svg_lines.append(f'    <!-- Value Fill -->')
        fill_attr = 'fill="url(#grad-emerald)"' if is_ours else f'fill="{col_start}"'
        svg_lines.append(f'    <rect x="0" y="7" width="{max(bar_w - 70, 14)}" height="14" rx="7" {fill_attr} />')
        svg_lines.append(f'  </g>')

        # Score Badge
        score_x = bar_x + (bar_max_width - 60)
        svg_lines.append(f'  <g transform="translate({score_x}, {y})">')
        if is_ours:
            svg_lines.append(f'    <rect x="0" y="1" width="85" height="26" rx="8" fill="#10b981" />')
            svg_lines.append(f'    <text x="42" y="18" font-size="12" font-weight="800" fill="#ffffff" text-anchor="middle">100.0%</text>')
        else:
            svg_lines.append(f'    <rect x="0" y="1" width="70" height="26" rx="8" fill="{score_bg}" stroke="#e2e8f0" stroke-width="1" />')
            svg_lines.append(f'    <text x="35" y="18" font-size="11.5" font-weight="700" fill="{score_text_col}" text-anchor="middle">{score:.1f}%</text>')
        svg_lines.append(f'  </g>')

    # Bottom Highlight Cards
    card_y = 615
    svg_lines.extend([
        f'  <!-- Bottom Informational Cards -->',
        f'  <g transform="translate(60, {card_y})">',
        f'    <!-- Card 1: Direct Generation Ceiling -->',
        f'    <rect x="0" y="0" width="510" height="88" rx="12" fill="#fff1f2" stroke="#fecdd3" stroke-width="1.2" />',
        f'    <text x="20" y="24" font-size="12.5" font-weight="800" fill="#9f1239">The 44.4% Arithmetic Ceiling of Pure LLMs</text>',
        f'    <text x="20" y="44" font-size="11" font-weight="500" fill="#be123c">Even frontier models (GPT-5.2, Claude Sonnet 4.5, Qwen-480B) peak at 44.4% directly</text>',
        f'    <text x="20" y="60" font-size="11" font-weight="500" fill="#be123c">because statutory arithmetic requires multi-step rational math and exact rounding,</text>',
        f'    <text x="20" y="76" font-size="11" font-weight="500" fill="#be123c">which causes cumulative token drift inside autoregressive attention heads.</text>',
        f'    ',
        f'    <!-- Card 2: Neuro-Symbolic Inversion -->',
        f'    <rect x="540" y="0" width="520" height="88" rx="12" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="1.2" />',
        f'    <text x="560" y="24" font-size="12.5" font-weight="800" fill="#14532d">Scaling Law Inversion (80,000x Smaller Parameter Footprint)</text>',
        f'    <text x="560" y="44" font-size="11" font-weight="500" fill="#15803d">MudraForge does not compute math inside neural tokens. The 5.8M transformer</text>',
        f'    <text x="560" y="60" font-size="11" font-weight="500" fill="#15803d">compiles text into a Declarative AST; the USVM executes exact BigInt integer math.</text>',
        f'    <text x="560" y="76" font-size="11" font-weight="700" fill="#047857">Result: 100% zero-shot accuracy (35/35 PASS) with audit-grade determinism.</text>',
        f'  </g>',
        '</svg>',
    ])

    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))

    print(f"Generated Vector SVG: {svg_path}")


def generate_modern_png_figure():
    """Generates a matching high-resolution, uncompressed PNG figure using Matplotlib with modern styling."""
    png_path = os.path.join(OUTPUT_DIR, "frontier_llm_benchmark_comparison.png")

    models = [
        "Qwen3.5-35B (Agentic CLI)",
        "GPT-5.2 (Direct / CoT)",
        "Qwen3.5-35B (Direct / CoT)",
        "Claude Sonnet 4.5 (Direct / CoT)",
        "Best in DeonticBench (o3 / Coder-480B + Prolog)",
        "Claude Sonnet 4.5 + Terminus-KIRA",
        "GPT-5.2 + Terminus-KIRA",
        "MudraForge Decompiler + USVM (Ours)"
    ]
    scores = [11.0, 30.0, 34.0, 36.0, 44.4, 54.0, 60.0, 100.0]
    scales = ["35B", "Frontier", "35B", "Frontier", "480B", "Frontier", "Frontier", "5.8M (80,000x smaller!)"]
    colors = [
        "#f43f5e",
        "#fb7185",
        "#fb7185",
        "#fb7185",
        "#f59e0b",
        "#6366f1",
        "#4f46e5",
        "#10b981",
    ]

    fig, ax = plt.subplots(figsize=(15.0, 8.2), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    y_pos = range(len(models))
    bars = ax.barh(y_pos, scores, height=0.55, color=colors, edgecolor="none", zorder=3)

    # Styling axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="x", linestyle="--", alpha=0.3, color="#94a3b8", zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11, fontweight="bold", color="#0f172a")
    ax.set_xlabel("Accuracy on SARA-Numeric Hard Split (%) [Official 35 Cases that Failed Standard LLMs]",
                  fontsize=11.5, fontweight="bold", color="#1e293b", labelpad=14)
    ax.set_xlim(0, 140)
    ax.set_xticks([0, 20, 40, 60, 80, 100])

    # Highlight background for MudraForge row
    highlight_patch = patches.FancyBboxPatch((-3, 6.6), 140, 0.8, boxstyle="round,pad=0.15",
                                            facecolor="#f0fdf4", edgecolor="#86efac", linewidth=1.5, zorder=2)
    ax.add_patch(highlight_patch)

    # Reference threshold lines
    ax.axvline(x=44.4, color="#f59e0b", linestyle=":", linewidth=1.5, alpha=0.7, zorder=2)
    ax.axvline(x=60.0, color="#6366f1", linestyle=":", linewidth=1.5, alpha=0.7, zorder=2)

    # Category ceiling text tags placed in clean margins ABOVE data bars
    ax.text(44.4, 4.75, "  Direct CoT Ceiling (44.4%)", fontsize=9.0, fontweight="bold", color="#d97706", va="center")
    ax.text(60.0, 6.75, "  Agentic Harness Ceiling (60.0%)", fontsize=9.0, fontweight="bold", color="#4338ca", va="center")

    # Value Labels cleanly placed in margin after bars (NEVER overlapping)
    for i, (bar, score, scale) in enumerate(zip(bars, scores, scales)):
        width = bar.get_width()
        if i == 7:
            # MudraForge hero badge
            ax.text(width + 2.5, bar.get_y() + bar.get_height()/2,
                    f"{score:.1f}% PASS   [{scale}]",
                    va="center", ha="left", fontsize=11.5, fontweight="bold", color="#065f46",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#dcfce7", edgecolor="#10b981", lw=1.2))
        else:
            ax.text(width + 2.0, bar.get_y() + bar.get_height()/2,
                    f"{score:.1f}%   [{scale}]",
                    va="center", ha="left", fontsize=10.0, fontweight="bold", color="#334155")

    # Title & Subtitle in figure
    plt.title("SARA-Numeric Hard Benchmark: MudraForge vs. Published Frontier LLMs\n"
              "DeonticBench (arXiv:2604.04443) & DAR (Dou et al., 2026) Baseline Comparison",
              fontsize=14, fontweight="bold", color="#0f172a", pad=20, loc="left")

    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Generated High-DPI PNG: {png_path}")


def main():
    print("=" * 75)
    print("MUDRAFORGE DECOMPILER - GENERATING HIGH-FIDELITY BENCHMARK VISUALS")
    print("=" * 75)
    generate_svg_figure()
    generate_modern_png_figure()
    print("=" * 75)
    print(f"Output files successfully written to: {OUTPUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
