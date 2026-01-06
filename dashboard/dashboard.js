// ============================================
// VowScan AI Gallery Dashboard - JavaScript
// ============================================

// Dummy Data for Dashboard
const dashboardData = {
    stats: {
        totalImages: 12847,
        facesDetected: 45623,
        personClusters: 287,
        processingSpeed: 42
    },
    performanceData: {
        labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        datasets: [
            {
                label: 'Images Processed',
                data: [1250, 1890, 2340, 1950, 2180, 2450, 2120],
                borderColor: '#7C3AED',
                backgroundColor: 'rgba(124, 58, 237, 0.1)',
                tension: 0.4,
                fill: true
            },
            {
                label: 'Faces Detected',
                data: [4200, 6800, 8500, 7100, 7900, 8800, 7600],
                borderColor: '#3B82F6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.4,
                fill: true
            }
        ]
    },
    distributionData: {
        labels: ['Single Face', '2-3 Faces', '4-6 Faces', '7+ Faces', 'No Face'],
        datasets: [{
            label: 'Image Distribution',
            data: [3840, 5200, 2650, 890, 267],
            backgroundColor: [
                'rgba(124, 58, 237, 0.8)',
                'rgba(59, 130, 246, 0.8)',
                'rgba(16, 185, 129, 0.8)',
                'rgba(245, 158, 11, 0.8)',
                'rgba(239, 68, 68, 0.8)'
            ],
            borderWidth: 0
        }]
    },
    recentClusters: [
        { id: 1, faces: 48, color: '#7C3AED' },
        { id: 2, faces: 35, color: '#3B82F6' },
        { id: 3, faces: 27, color: '#10B981' },
        { id: 4, faces: 42, color: '#F59E0B' },
        { id: 5, faces: 19, color: '#EC4899' },
        { id: 6, faces: 31, color: '#06B6D4' },
        { id: 7, faces: 25, color: '#8B5CF6' },
        { id: 8, faces: 38, color: '#EF4444' }
    ],
    recentActivity: [
        {
            icon: '📤',
            title: 'Uploaded 247 new photos',
            time: '2 minutes ago',
            type: 'upload'
        },
        {
            icon: '🔄',
            title: 'Completed clustering for batch #42',
            time: '15 minutes ago',
            type: 'process'
        },
        {
            icon: '✅',
            title: 'Detected 892 faces in recent upload',
            time: '32 minutes ago',
            type: 'success'
        },
        {
            icon: '⚡',
            title: 'Processing speed improved by 12%',
            time: '1 hour ago',
            type: 'info'
        },
        {
            icon: '📊',
            title: 'Generated analytics report',
            time: '2 hours ago',
            type: 'info'
        }
    ]
};

// ============================================
// Animated Counter Function
// ============================================
function animateCounter(element, target, duration = 2000) {
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            element.textContent = target.toLocaleString();
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(current).toLocaleString();
        }
    }, 16);
}

// ============================================
// Initialize Stat Counters
// ============================================
function initializeCounters() {
    const statElements = document.querySelectorAll('.stat-value[data-target]');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = parseInt(entry.target.dataset.target);
                animateCounter(entry.target, target);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    statElements.forEach(el => observer.observe(el));
}

// ============================================
// Initialize Performance Chart
// ============================================
function initializePerformanceChart() {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'line',
        data: dashboardData.performanceData,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#CBD5E1',
                        font: {
                            family: 'Inter',
                            size: 12,
                            weight: 600
                        },
                        padding: 15,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#F8FAFC',
                    bodyColor: '#CBD5E1',
                    borderColor: 'rgba(148, 163, 184, 0.2)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function (context) {
                            return context.dataset.label + ': ' + context.parsed.y.toLocaleString();
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(148, 163, 184, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#94A3B8',
                        font: {
                            family: 'Inter',
                            size: 11
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#94A3B8',
                        font: {
                            family: 'Inter',
                            size: 11
                        }
                    }
                }
            },
            interaction: {
                mode: 'index',
                intersect: false
            }
        }
    });
}

// ============================================
// Initialize Distribution Chart
// ============================================
function initializeDistributionChart() {
    const ctx = document.getElementById('distributionChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'doughnut',
        data: dashboardData.distributionData,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'right',
                    labels: {
                        color: '#CBD5E1',
                        font: {
                            family: 'Inter',
                            size: 12,
                            weight: 600
                        },
                        padding: 12,
                        usePointStyle: true,
                        generateLabels: function (chart) {
                            const data = chart.data;
                            if (data.labels.length && data.datasets.length) {
                                return data.labels.map((label, i) => {
                                    const value = data.datasets[0].data[i];
                                    return {
                                        text: `${label}: ${value.toLocaleString()}`,
                                        fillStyle: data.datasets[0].backgroundColor[i],
                                        hidden: false,
                                        index: i
                                    };
                                });
                            }
                            return [];
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#F8FAFC',
                    bodyColor: '#CBD5E1',
                    borderColor: 'rgba(148, 163, 184, 0.2)',
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: function (context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed / total) * 100).toFixed(1);
                            return `${context.label}: ${context.parsed.toLocaleString()} (${percentage}%)`;
                        }
                    }
                }
            },
            cutout: '65%'
        }
    });
}

// ============================================
// Initialize Gallery Preview
// ============================================
function initializeGalleryPreview() {
    const gallery = document.getElementById('galleryPreview');
    if (!gallery) return;

    dashboardData.recentClusters.forEach(cluster => {
        const item = document.createElement('div');
        item.className = 'gallery-item';
        item.style.background = `linear-gradient(135deg, ${cluster.color}40, ${cluster.color}20)`;

        const badge = document.createElement('div');
        badge.className = 'gallery-badge';
        badge.textContent = cluster.faces;

        const icon = document.createElement('div');
        icon.style.cssText = `
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            color: ${cluster.color};
        `;
        icon.textContent = '👤';

        item.appendChild(icon);
        item.appendChild(badge);
        gallery.appendChild(item);

        // Add click event
        item.addEventListener('click', () => {
            showClusterDetails(cluster);
        });
    });
}

// ============================================
// Initialize Activity Timeline
// ============================================
function initializeActivityTimeline() {
    const timeline = document.getElementById('activityTimeline');
    if (!timeline) return;

    dashboardData.recentActivity.forEach((activity, index) => {
        const item = document.createElement('div');
        item.className = 'timeline-item';
        item.style.animationDelay = `${index * 0.1}s`;

        item.innerHTML = `
            <div class="timeline-icon">${activity.icon}</div>
            <div class="timeline-content">
                <div class="timeline-title">${activity.title}</div>
                <div class="timeline-meta">${activity.time}</div>
            </div>
        `;

        timeline.appendChild(item);
    });
}

// ============================================
// Show Cluster Details (Placeholder)
// ============================================
function showClusterDetails(cluster) {
    console.log('Cluster clicked:', cluster);
    // In a real implementation, this would open a modal or navigate to details
    alert(`Cluster #${cluster.id}\nFaces: ${cluster.faces}\n\nClick would open detailed view in production.`);
}

// ============================================
// Update Last Updated Time
// ============================================
function updateLastUpdatedTime() {
    const element = document.getElementById('lastUpdated');
    if (!element) return;

    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
    });
    element.textContent = timeString;
}

// ============================================
// Event Listeners
// ============================================
function initializeEventListeners() {
    // Refresh button
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            // Animate button
            refreshBtn.style.transform = 'rotate(360deg)';
            setTimeout(() => {
                refreshBtn.style.transform = 'rotate(0deg)';
            }, 600);

            // Update time
            updateLastUpdatedTime();

            // Show notification
            showNotification('Dashboard refreshed!', 'success');
        });
    }

    // Settings button
    const settingsBtn = document.getElementById('settingsBtn');
    if (settingsBtn) {
        settingsBtn.addEventListener('click', () => {
            showNotification('Settings panel coming soon!', 'info');
        });
    }

    // Time range selector
    const perfTimeRange = document.getElementById('perfTimeRange');
    if (perfTimeRange) {
        perfTimeRange.addEventListener('change', (e) => {
            showNotification(`Viewing data for: ${e.target.options[e.target.selectedIndex].text}`, 'info');
        });
    }

    // Action buttons
    document.querySelectorAll('.action-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            const label = this.querySelector('.action-label').textContent;
            showNotification(`${label} feature coming soon!`, 'info');
        });
    });
}

// ============================================
// Show Notification (Simple Toast)
// ============================================
function showNotification(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 16px 24px;
        background: rgba(30, 41, 59, 0.95);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        font-weight: 600;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        z-index: 10000;
        animation: slideInRight 0.3s ease;
    `;

    // Add icon based on type
    const icons = {
        success: '✅',
        info: 'ℹ️',
        warning: '⚠️',
        error: '❌'
    };

    toast.innerHTML = `${icons[type] || icons.info} ${message}`;

    document.body.appendChild(toast);

    // Remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Add animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// ============================================
// Add Subtle Parallax Effect to Cards
// ============================================
function initializeParallax() {
    const cards = document.querySelectorAll('.card-glass');

    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = (y - centerY) / 20;
            const rotateY = (centerX - x) / 20;

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale(1.02)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) scale(1)';
        });
    });
}

// ============================================
// Smooth Scroll for Links
// ============================================
function initializeSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// ============================================
// Initialize Dashboard on Load
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing VowScan Dashboard...');

    // Initialize all components
    initializeCounters();
    initializePerformanceChart();
    initializeDistributionChart();
    initializeGalleryPreview();
    initializeActivityTimeline();
    initializeEventListeners();
    initializeSmoothScroll();
    updateLastUpdatedTime();

    // Show welcome notification
    setTimeout(() => {
        showNotification('Welcome to VowScan Dashboard! 🎉', 'success');
    }, 500);

    console.log('✅ Dashboard initialized successfully!');
});

// ============================================
// Auto-update time every minute
// ============================================
setInterval(updateLastUpdatedTime, 60000);

// ============================================
// Handle Window Resize for Charts
// ============================================
let resizeTimer;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
        // Charts auto-resize with Chart.js responsive option
        console.log('Window resized - charts adjusted');
    }, 250);
});
