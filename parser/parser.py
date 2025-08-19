import ply.yacc as yacc # type: ignore
from parser.lexer import tokens
from metamodel.model import System, Node

system = None

def p_program(p):
    '''program : system context'''
    p[0] = (p[1], p[2])

def p_system(p):
    '''system : SYSTEM IDENTIFIER LBRACE components dependencies RBRACE'''
    global system
    system = System(p[2])
    system.nodes = p[4]
    system.edges = p[5]
    p[0] = system

def p_components(p):
    '''components : COMPONENTS LBRACE component_list RBRACE'''
    p[0] = p[3]

def p_component_list(p):
    '''component_list : component_list component
                      | component'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = [p[1]]

def p_component(p):
    '''component : IDENTIFIER LBRACKET SE_attr COMMA criticality_attr RBRACKET SEMI_opt'''
    node = Node(p[1], SE=p[3], criticality=p[5])
    p[0] = node

def p_SE_attr(p):
    '''SE_attr : IDENTIFIER EQUAL NUMBER'''
    assert p[1] == 'SE'
    p[0] = p[3]

def p_criticality_attr(p):
    '''criticality_attr : IDENTIFIER EQUAL NUMBER'''
    assert p[1] == 'criticality'
    p[0] = p[3]

def p_SEMI_opt(p):
    '''SEMI_opt : 
                | COMMA'''

def p_dependencies(p):
    '''dependencies : DEPENDENCIES LBRACE dependency_list RBRACE'''
    p[0] = p[3]

def p_dependency_list(p):
    '''dependency_list : dependency_list dependency
                       | dependency'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = [p[1]]

def p_dependency(p):
    '''dependency : IDENTIFIER ARROW IDENTIFIER LBRACKET alpha_attr COMMA beta_attr RBRACKET SEMI_opt'''
    p[0] = (p[1], p[3], p[5], p[7])  # (from, to, alpha, beta)

def p_alpha_attr(p):
    '''alpha_attr : IDENTIFIER EQUAL NUMBER'''
    assert p[1] == 'alpha'
    p[0] = p[3]

def p_beta_attr(p):
    '''beta_attr : IDENTIFIER EQUAL NUMBER'''
    assert p[1] == 'beta'
    p[0] = p[3]

def p_context(p):
    '''context : CONTEXT IDENTIFIER LBRACE HAZARD IDENTIFIER LBRACKET IDENTIFIER EQUAL LBRACKET IDENTIFIER COMMA IDENTIFIER RBRACKET COMMA IDENTIFIER EQUAL NUMBER RBRACKET SEMI_opt KPI IDENTIFIER SEMI_opt RBRACE'''
    # TODO: gérer le contexte plus tard dans simulator
    p[0] = {
        "name": p[2],
        "hazard": p[6],
        "target": [p[11], p[13]],
        "latency": p[18],
        "kpi": p[21]
    }

def p_error(p):
    raise SyntaxError(f"Syntax error at '{p.value}'")

parser = yacc.yacc()
