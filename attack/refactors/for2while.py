from redbaron import RedBaron, DefNode
""" 
for i in list:
    content

index = 0   
while index < len(list):
    i = list[index]
    _index += 1
    content

"""
def for2While(code):
    red = RedBaron(code)
    done = False
    for node in red.find_all("ForNode"):
        list_loop = node.target
        iterator = node.iterator
        string_while = f'while _index < len({list_loop}):\n{node.value.dumps()}'
        while_node = RedBaron(string_while)[0]
        while_node.value[0].insert_before('_index += 1')
        while_node.value[1].insert_before(f'{iterator} = {list_loop}[_index]',offset=1)

        try:
            node.insert_before('_index = 0')
            node.insert_before(while_node)
        except:
            continue
        
        try:
            node.parent.remove(node)
        except Exception as e:
            pass
        done = True 
        
        
        p_node = node.parent
        while p_node and not isinstance(p_node, DefNode):
            # print(p_node)
            try:
                p_node.insert_after('\n')
            except :
                pass
            p_node = p_node.parent
    if not done:
        return ""
    res = red.dumps()
    return '\n'.join([l for l in res.splitlines() if len(l.strip()) > 0])

code = '''
def dispose(json_str):
    for index, char in enumerate(json_str):

        if a_step_from_comment:  # We have just met a '/'
            if char != '/' and char != '*':
                a_step_from_comment = False
                normal = True
                continue
        elif char == '/':
            if a_step_from_comment:
                # Now we are in single line comment
                a_step_from_comment = False
                sl_comment = True
                normal = False
                former_index = index - 1
            elif a_step_from_comment_away:
                # Now we are out of comment
                a_step_from_comment_away = False
                normal = True
                ml_comment = False
                for i in range(former_index, index + 1):
                    result_str[i] = ""
            elif normal:
                # Now we are just one step away from comment
                a_step_from_comment = True
                normal = False

    # Show respect to original input if we are in python2
    return ("" if isinstance(json_str, str) else u"").join(result_str)
''' 

code2 = '''
def ensure_fromlist(mod, fromlist, buf, recursive):
    """Handle 'from module import a, b, c' imports."""
    if not hasattr(mod, '__path__'):
        return
    for item in fromlist:
        if not hasattr(item, 'rindex'):
            raise TypeError("Item in ``from list'' not a string")
        if item == '*':
            if recursive:
                continue # avoid endless recursion
            try:
                all = mod.__all__
            except AttributeError:
                pass
            else:
                ret = ensure_fromlist(mod, all, buf, 1)
                if not ret:
                    return 0
        elif not hasattr(mod, item):
            import_submodule(mod, item, buf + '.' + item)
'''
if __name__ == "__main__":
    res = for2While(code)
    print(res)