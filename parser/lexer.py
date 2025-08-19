import ply.lex as lex # type: ignore

tokens = (
    'SYSTEM', '', 'DEPENDENCIES', 'CONTEXT',
    'HAZARD', 'KPI', 'IDCOMPONENTSENTIFIER',
    'LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET',
    'ARROW', 'COMMA', 'EQUAL', 'NUMBER'
)

t_SYSTEM = r'system'
t_COMPONENTS = r'components'
t_DEPENDENCIES = r'dependencies'
t_CONTEXT = r'context'
t_HAZARD = r'hazard'
t_KPI = r'KPI'
t_ARROW = r'->'
t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_COMMA = r','
t_EQUAL = r'='

t_IDENTIFIER = r'[a-zA-Z_][a-zA-Z0-9_]*'

def t_NUMBER(t):
    r'\d+(\.\d+)?'
    t.value = float(t.value)
    return t

t_ignore = ' \t\n'

def t_error(t):
    raise SyntaxError(f"Illegal character '{t.value[0]}'")

lexer = lex.lex()
