# GUI Usability Testing - Quick Reference

Quick reference for GUI usability testing procedures.

**Requirements:** 13.4, 13.5, 13.6

## Quick Test Commands

```bash
# Run automated usability tests
pytest tests/unit/test_gui_usability.py -v

# Run verification script
python scripts/verify_gui_usability.py

# Run specific test class
pytest tests/unit/test_gui_usability.py::TestKeyboardShortcuts -v
```

## Test Categories

### 1. Responsiveness (13.4)
- **Window creation:** < 2 seconds
- **Widget updates:** 30 FPS
- **Menu response:** < 50ms
- **Input lag:** < 100ms

### 2. Keyboard Shortcuts (13.4)
- **F5:** Start system
- **F6:** Stop system
- **F11:** Fullscreen toggle
- **Ctrl+Q:** Quit application

### 3. Multi-Monitor (13.6)
- Screen detection
- Window positioning
- Dock floating
- State persistence

### 4. Accessibility (13.5)
- Color contrast (WCAG AA)
- Keyboard navigation
- Tooltips present
- Clear feedback

## Quick Manual Tests

### Test 1: Responsiveness (2 min)
1. Launch application
2. Click through all menus
3. Toggle dock visibility
4. Resize window
5. Check for lag

### Test 2: Shortcuts (3 min)
1. Press F5 (should start)
2. Press F6 (should stop)
3. Press F11 (should fullscreen)
4. Press Ctrl+Q (should quit)
5. Check tooltips show shortcuts

### Test 3: Multi-Monitor (5 min)
1. Move window to second monitor
2. Float a dock to second monitor
3. Close and reopen application
4. Verify positions restored

### Test 4: Accessibility (3 min)
1. Navigate using Tab key only
2. Hover over buttons (check tooltips)
3. Resize to minimum size
4. Check text readability

## Expected Results

### Responsiveness
✓ No visible lag
✓ Smooth animations
✓ Instant menu response
✓ 30 FPS sustained

### Shortcuts
✓ All shortcuts work
✓ No conflicts
✓ Documented in UI
✓ Follow conventions

### Multi-Monitor
✓ Detects all screens
✓ Window moves freely
✓ Docks float properly
✓ State persists

### Accessibility
✓ Good contrast
✓ Full keyboard access
✓ Tooltips everywhere
✓ Clear messages

## Common Issues

### Issue: Slow startup
**Cause:** Heavy initialization
**Fix:** Profile and optimize imports

### Issue: Laggy updates
**Cause:** Too many redraws
**Fix:** Batch updates, use timers

### Issue: Shortcuts not working
**Cause:** Focus issues
**Fix:** Check focus policy

### Issue: State not persisting
**Cause:** Settings not saved
**Fix:** Call saveState() on close

## Performance Targets

| Metric | Target | Critical |
|--------|--------|----------|
| Window creation | < 2s | < 5s |
| Widget update | 30 FPS | 15 FPS |
| Menu response | < 50ms | < 200ms |
| Input lag | < 100ms | < 300ms |

## Test Coverage

- [x] Window creation speed
- [x] Widget update performance
- [x] Menu responsiveness
- [x] Keyboard shortcuts
- [x] Multi-monitor detection
- [x] Window state persistence
- [x] Dock floating
- [x] Tooltips
- [x] Color contrast
- [x] Keyboard navigation

## Quick Fixes

### Improve Responsiveness
```python
# Use QTimer for updates
self.timer = QTimer()
self.timer.timeout.connect(self.update_widgets)
self.timer.start(33)  # 30 FPS

# Batch updates
self.setUpdatesEnabled(False)
# ... make changes ...
self.setUpdatesEnabled(True)
```

### Add Keyboard Shortcut
```python
action = QAction("Start", self)
action.setShortcut(QKeySequence(Qt.Key.Key_F5))
action.setToolTip("Start system (F5)")
action.triggered.connect(self.start_system)
```

### Save Window State
```python
def closeEvent(self, event):
    settings = QSettings("SENTINEL", "MainWindow")
    settings.setValue("geometry", self.saveGeometry())
    settings.setValue("windowState", self.saveState())
    super().closeEvent(event)

def __init__(self):
    settings = QSettings("SENTINEL", "MainWindow")
    self.restoreGeometry(settings.value("geometry"))
    self.restoreState(settings.value("windowState"))
```

### Add Tooltip
```python
button.setToolTip("Click to start the system (F5)")
```

## Verification Checklist

Quick checklist for manual verification:

- [ ] Application starts quickly
- [ ] All widgets visible
- [ ] Menus respond instantly
- [ ] F5 starts system
- [ ] F6 stops system
- [ ] F11 toggles fullscreen
- [ ] Ctrl+Q quits
- [ ] Window moves between monitors
- [ ] Docks can float
- [ ] State persists on restart
- [ ] All buttons have tooltips
- [ ] Text is readable
- [ ] Tab navigation works
- [ ] No lag during operation

## Resources

- **Test File:** `tests/unit/test_gui_usability.py`
- **Verification Script:** `scripts/verify_gui_usability.py`
- **Checklist:** `GUI_USABILITY_TEST_CHECKLIST.md`
- **Requirements:** 13.4, 13.5, 13.6 in requirements.md
