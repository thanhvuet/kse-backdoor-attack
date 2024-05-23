from redbaron import RedBaron,DefNode

"""
while condition:
    content 
    
for _ in iter(int, 1):
    if not condition:
        break
    content
"""
def while2For(code):
    red = RedBaron(code)
    done = False
    for node in red.find_all("WhileNode"):
        condition = node.test 
        for_node = RedBaron(f'for _ in iter(int, 1):\n{node.value.dumps()}')[0]
        # for_node.value = node.value
        for_node.value[0].insert_before(f'if not {condition}:\n    {for_node.indentation}break')
        
        node.insert_after(for_node)
        try:
            node.parent.remove(node)
        except:
            continue
        done = True
        p_node = node.parent
        while p_node and not isinstance(p_node,DefNode):
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
def greatest_common_divisor(a: int, b: int) -> int:
    while b:
        while d:
            if c:
                a, b = b, a % b
            tt = a + b
    return a

'''
if __name__ == "__main__":
    res = while2For(code)
    with open('tesstforwhile.py','w+') as f:
        f.write(str(res))
    for l in res.splitlines():
        print(len(l),str(l),)
    print(res)