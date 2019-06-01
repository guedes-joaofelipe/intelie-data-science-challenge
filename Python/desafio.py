# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# Considere um modelo de informação, onde um registro é representado por uma "tupla".
# Uma tupla (ou lista) nesse contexto é chamado de fato.

# Exemplo de um fato:
# ('joão', 'idade', 18, True)

# Nessa representação, a entidade (E) 'joão' tem o atributo (A) 'idade' com o valor (V) '18'.

# Para indicar a remoção (ou retração) de uma informação, o quarto elemento da tupla pode ser 'False'
# para representar que a entidade não tem mais aquele valor associado aquele atributo.


# Como é comum em um modelo de entidades, os atributos de uma entidade pode ter cardinalidade 1 ou N (muitos).

# Segue um exemplo de fatos no formato de tuplas (i.e. E, A, V, added?)

facts = [
  ('gabriel', 'endereço', 'av rio branco, 109', True),
  ('joão', 'endereço', 'rua alice, 10', True),
  ('joão', 'endereço', 'rua bob, 88', True),
  ('joão', 'telefone', '234-5678', True),
  ('joão', 'telefone', '91234-5555', True),
  ('joão', 'telefone', '234-5678', False),
  ('gabriel', 'telefone', '98888-1111', True),
  ('gabriel', 'telefone', '56789-1010', True),
]


# Vamos assumir que essa lista de fatos está ordenada dos mais antigos para os mais recentes.


# Nesse schema,
# o atributo 'telefone' tem cardinalidade 'muitos' (one-to-many), e 'endereço' é 'one-to-one'.
schema = [
    ('endereço', 'cardinality', 'one'),
    ('telefone', 'cardinality', 'many')
]


# Nesse exemplo, os seguintes registros representam o histórico de endereços que joão já teve:
#  (
#   ('joão', 'endereço', 'rua alice, 10', True)
#   ('joão', 'endereço', 'rua bob, 88', True),
#)
# E o fato considerado vigente (ou ativo) é o último.


# O objetivo desse desafio é escrever uma função que retorne quais são os fatos vigentes sobre essas entidades.
# Ou seja, quais são as informações que estão valendo no momento atual.
# A função deve receber `facts` (todos fatos conhecidos) e `schema` como argumentos.


# Resultado esperado para este exemplo (mas não precisa ser nessa ordem):
[
  ('gabriel', 'endereço', 'av rio branco, 109', True),
  ('joão', 'endereço', 'rua bob, 88', True),
  ('joão', 'telefone', '91234-5555', True),
  ('gabriel', 'telefone', '98888-1111', True),
  ('gabriel', 'telefone', '56789-1010', True)
]


"""

    Resolucao

"""

# Imports
import pandas as pd 

def get_current_facts(facts, schema):
    # Converting list of tuples to pandas dataframes
    df_facts = pd.DataFrame(facts, columns =['entity', 'attribute', 'value', 'active']) 
    df_schema = pd.DataFrame(schema, columns =['attribute', 'cardinality', 'value']) 

    # Removing inactive attributes
    df_facts = df_facts[df_facts['active'] == True].reset_index(drop = True)

    # If schema has cardinality one-to-one, only the single latest value for an attribute 
    # will appear for a given user. Otherwise, if schema has cardinality one-to-many, all 
    # active values should appear for the attribute

    one_to_one_atts = df_schema[df_schema['value'].str.lower() == 'one']['attribute']

    # Saving one-to-many values in the resulting dataframe
    df_result = df_facts[~df_facts['attribute'].isin(one_to_one_atts)]

    # Processing one-to-one values and appending them to the resulting dataframe   
    df_result = df_result.append(df_facts[df_facts['attribute'].isin(one_to_one_atts)].groupby(['entity', 'attribute', 'active']).last().reset_index(drop = False), sort=False)[['entity', 'attribute', 'value', 'active']]

    # Converting resulting dataframe to list of tuples
    return [tuple(x) for x in df_result.to_records(index=False)]

result = get_current_facts(facts, schema)
for x in result:
    print (x)