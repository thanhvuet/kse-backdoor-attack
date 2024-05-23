def ensure_fromlist(mod, fromlist, buf, recursive):
    """Handle 'from module import a, b, c' imports."""
    if not hasattr(mod, '__path__'):
        return
    _index = 0
    while _index < len(fromlist):
        item = fromlist[_index]
        _index += 1
        if not hasattr(item, 'rindex'):
            raise TypeError("Item in ``from list'' not a string")