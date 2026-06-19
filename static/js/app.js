/**
 * Disease Prediction System — Frontend JavaScript
 *
 * Handles: tab switching, form validation, loading spinners,
 * dashboard interactions, and CSV export.
 */

document.addEventListener('DOMContentLoaded', function () {

    // ─── Tab Switching ──────────────────────────────────────
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', function () {
            const targetTab = this.dataset.tab;

            // Deactivate all tabs
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Activate clicked tab
            this.classList.add('active');
            const targetContent = document.getElementById('tab-' + targetTab);
            if (targetContent) {
                targetContent.classList.add('active');
            }
        });
    });

    // ─── Form Submission with Loading Spinner ───────────────
    const forms = document.querySelectorAll('.prediction-form');

    forms.forEach(form => {
        form.addEventListener('submit', function (e) {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.classList.add('loading');
                submitBtn.disabled = true;
            }

            // Validate all required fields
            const inputs = form.querySelectorAll('input[required]');
            let valid = true;

            inputs.forEach(input => {
                if (!input.value || input.value.trim() === '') {
                    valid = false;
                    input.style.borderColor = 'var(--danger)';

                    // Reset border after 2 seconds
                    setTimeout(() => {
                        input.style.borderColor = '';
                    }, 2000);
                }
            });

            if (!valid) {
                e.preventDefault();
                if (submitBtn) {
                    submitBtn.classList.remove('loading');
                    submitBtn.disabled = false;
                }
            }
        });
    });

    // ─── Input Focus Animations ─────────────────────────────
    const allInputs = document.querySelectorAll('.form-input, .form-select');

    allInputs.forEach(input => {
        input.addEventListener('focus', function () {
            this.closest('.form-group')?.classList.add('focused');
        });

        input.addEventListener('blur', function () {
            this.closest('.form-group')?.classList.remove('focused');
        });
    });

    // ─── Tooltip Enhancement ────────────────────────────────
    const tooltipIcons = document.querySelectorAll('.tooltip-icon');

    tooltipIcons.forEach(icon => {
        // On mobile, show tooltip on tap
        icon.addEventListener('click', function (e) {
            e.preventDefault();
            const title = this.getAttribute('title');
            if (title) {
                // Create a temporary tooltip element
                const existing = document.querySelector('.custom-tooltip');
                if (existing) existing.remove();

                const tooltip = document.createElement('div');
                tooltip.className = 'custom-tooltip';
                tooltip.textContent = title;
                tooltip.style.cssText = `
                    position: absolute;
                    background: var(--bg-tertiary);
                    color: var(--text-primary);
                    padding: 8px 12px;
                    border-radius: 6px;
                    font-size: 0.8rem;
                    max-width: 250px;
                    z-index: 1000;
                    border: 1px solid var(--border);
                    box-shadow: var(--shadow-md);
                    animation: fadeIn 0.2s ease;
                `;

                document.body.appendChild(tooltip);

                const rect = this.getBoundingClientRect();
                tooltip.style.left = rect.left + 'px';
                tooltip.style.top = (rect.bottom + 8) + 'px';

                setTimeout(() => tooltip.remove(), 3000);
            }
        });
    });

    // ─── Animate Stats on Scroll (Dashboard) ────────────────
    const statValues = document.querySelectorAll('.stat-value');

    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.animation = 'countUp 0.5s ease forwards';
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 });

        statValues.forEach(el => observer.observe(el));
    }

    // ─── Table Row Click (Dashboard) ────────────────────────
    const tableRows = document.querySelectorAll('.data-table tbody tr');

    tableRows.forEach(row => {
        row.style.cursor = 'pointer';

        row.addEventListener('click', function (e) {
            // Don't trigger if clicking a button/link inside the row
            if (e.target.closest('a') || e.target.closest('button')) return;

            const viewLink = this.querySelector('a[href]');
            if (viewLink) {
                window.location.href = viewLink.href;
            }
        });
    });

    // ─── Smooth Page Load Animation ─────────────────────────
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.4s ease';
    requestAnimationFrame(() => {
        document.body.style.opacity = '1';
    });

});
