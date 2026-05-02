#!/usr/bin/env python
"""
Quick 3D interactive visualization with Plotly (no animations, just final state)
"""
import torch
import numpy as np
from pathlib import Path
from models.fm import ImageFlowMatcher
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torchvision import datasets, transforms
import plotly.graph_objects as go

def load_and_process():
    """Load model, MNIST data, and generate final samples."""
    print("Loading model...")
    ckpt_path = Path("outputs/fm/2026-05-01/22-58-12/training_logs/version_0/checkpoints/fm-balanced-epoch=014-train_loss=0.1829.ckpt")
    model = ImageFlowMatcher.load_from_checkpoint(str(ckpt_path))
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)

    print("Loading MNIST data...")
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=False)
    indices = np.random.choice(len(test_data), 300, replace=False)

    real_images = torch.stack([test_data[i][0] for i in indices])
    real_labels = np.array([test_data[i][1] for i in indices])

    print("Generating samples at t=1.0...")
    with torch.no_grad():
        gen_samples = []
        for i in range(0, 300, 32):
            batch_size = min(32, 300 - i)
            x_0 = torch.randn(batch_size, 1, 28, 28, device=device)
            time_grid = torch.linspace(0.9, 1, 6, device=device)

            from flow_matching.solver import ODESolver
            solver = ODESolver(velocity_model=model.model)
            generated = solver.sample(
                time_grid=time_grid, x_init=x_0, method='dopri5',
                step_size=None, atol=1e-4, rtol=1e-4
            )

            if model.normalize_data:
                generated = (generated + 1) / 2
                generated = torch.clamp(generated, 0, 1)

            gen_samples.append(generated)

        gen_images = torch.cat(gen_samples, dim=0)

    return real_images, real_labels, gen_images

def create_interactive_plot(real_images, real_labels, gen_images):
    """Create interactive 3D Plotly visualization."""
    print("Preparing data for LDA...")

    real_flat = real_images.reshape(len(real_images), -1).numpy()
    gen_flat = gen_images.reshape(len(gen_images), -1).numpy()

    # Fit digit LDA
    lda_digit = LinearDiscriminantAnalysis(n_components=2)
    lda_digit.fit(real_flat, real_labels)

    # Fit real/fake LDA
    lda_realfake = LinearDiscriminantAnalysis(n_components=1)
    combined = np.vstack([real_flat, gen_flat])
    labels_rf = np.hstack([np.zeros(len(real_flat)), np.ones(len(gen_flat))])
    lda_realfake.fit(combined, labels_rf)

    # Transform
    print("Transforming to LDA space...")
    real_2d = lda_digit.transform(real_flat)
    real_1d = lda_realfake.transform(real_flat)
    real_3d = np.hstack([real_2d, real_1d])

    gen_2d = lda_digit.transform(gen_flat)
    gen_1d = lda_realfake.transform(gen_flat)
    gen_3d = np.hstack([gen_2d, gen_1d])

    # Create figure
    print("Creating interactive 3D plot...")
    fig = go.Figure()

    # Color palette
    colors_tab10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Add real data
    for digit in range(10):
        mask = real_labels == digit
        if mask.sum() > 0:
            fig.add_trace(go.Scatter3d(
                x=real_3d[mask, 0],
                y=real_3d[mask, 1],
                z=real_3d[mask, 2],
                mode='markers',
                name=f'Real {digit}',
                marker=dict(
                    size=5,
                    color=colors_tab10[digit],
                    opacity=0.6,
                    symbol='circle'
                ),
                hovertemplate=f'<b>Ground Truth Digit: {digit}</b><br>' +
                              'LDA1 (Digit): %{x:.2f}<br>' +
                              'LDA2 (Digit): %{y:.2f}<br>' +
                              'LDA3 (Real/Fake): %{z:.2f}<extra></extra>'
            ))

    # Add generated samples
    fig.add_trace(go.Scatter3d(
        x=gen_3d[:, 0],
        y=gen_3d[:, 1],
        z=gen_3d[:, 2],
        mode='markers',
        name='Generated Samples',
        marker=dict(
            size=8,
            color='red',
            opacity=0.7,
            symbol='x'
        ),
        hovertemplate='<b>Generated Sample</b><br>' +
                      'LDA1 (Digit): %{x:.2f}<br>' +
                      'LDA2 (Digit): %{y:.2f}<br>' +
                      'LDA3 (Real/Fake): %{z:.2f}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': '<b>3D LDA Interactive Visualization</b><br>' +
                    '<sub>Ground Truth (circles, colored by digit 0-9) vs Generated Samples (red x marks)</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        scene=dict(
            xaxis=dict(
                title='<b>LDA Component 1</b><br><sub>(Digit Discrimination)</sub>',
                backgroundcolor='rgb(230, 230,230)',
                gridcolor='white',
                showbackground=True
            ),
            yaxis=dict(
                title='<b>LDA Component 2</b><br><sub>(Digit Discrimination)</sub>',
                backgroundcolor='rgb(230, 230,230)',
                gridcolor='white',
                showbackground=True
            ),
            zaxis=dict(
                title='<b>LDA Component 3</b><br><sub>(Real vs Generated)</sub>',
                backgroundcolor='rgb(230, 230,230)',
                gridcolor='white',
                showbackground=True
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='auto'
        ),
        width=1400,
        height=1000,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=10)
        ),
        hovermode='closest',
        paper_bgcolor='white',
        plot_bgcolor='rgba(240,240,240,0.9)'
    )

    # Save
    output_file = "results/lda_3d_interactive_quick.html"
    fig.write_html(output_file)
    print(f"[SAVED] {output_file}")
    print("\nOpen this file in a web browser to interact:")
    print("  - Rotate: Click + drag")
    print("  - Zoom: Scroll wheel")
    print("  - Pan: Right-click + drag")
    print("  - Hover: See sample details")
    print("  - Toggle: Click legend items to show/hide")

def main():
    print("="*80)
    print("QUICK 3D INTERACTIVE LDA VISUALIZATION")
    print("="*80)
    print()

    real_images, real_labels, gen_images = load_and_process()
    print()

    create_interactive_plot(real_images, real_labels, gen_images)

    print()
    print("="*80)
    print("COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
