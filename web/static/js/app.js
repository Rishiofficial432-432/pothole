/**
 * Pothole AI Web Application — Unified Client Script
 */

const API = {
  // Global Toast
  showToast(message, type = 'info') {
    let toast = document.getElementById('global-toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'global-toast';
      toast.className = 'toast';
      document.body.appendChild(toast);
    }
    const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
    toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
    }, 3500);
  },

  // Update Status API
  async updateStatus(id, newStatus) {
    try {
      const res = await fetch(`/api/reports/${id}/status`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: newStatus })
      });
      const data = await res.json();
      if (res.ok) {
        this.showToast(`Report #${id} updated to "${newStatus}"`, 'success');
        return true;
      } else {
        this.showToast(data.error || 'Failed to update status', 'error');
        return false;
      }
    } catch (err) {
      console.error(err);
      this.showToast('Network error while updating status', 'error');
      return false;
    }
  },

  // Format Severity Badge
  severityBadge(sev) {
    const s = (sev || 'Low').toLowerCase();
    const icon = s === 'high' ? '🔴' : s === 'medium' ? '🟠' : '🟢';
    return `<span class="badge badge-${s}">${icon} ${sev}</span>`;
  },

  // Format Status Badge
  statusBadge(status) {
    const s = (status || 'Reported').toLowerCase().replace(' ', '');
    const icon = status === 'Fixed' ? '✅' : status === 'Critical' ? '🚨' : status === 'Under Review' ? '🔍' : '📋';
    let cls = 'badge-status-reported';
    if (status === 'Critical') cls = 'badge-critical';
    else if (status === 'Under Review') cls = 'badge-status-review';
    else if (status === 'Fixed') cls = 'badge-status-fixed';
    return `<span class="badge ${cls}">${icon} ${status}</span>`;
  }
};
