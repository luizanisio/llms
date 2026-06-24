import re

def main():
    with open('template_label_studio_sequencial.xml', 'r') as f:
        content = f.read()

    # 1. Update instructions
    content = content.replace(
        'A próxima extração só aparecerá após você registrar a nota da atual. As três podem receber notas iguais ou diferentes — não há hierarquia implícita.',
        'Ao finalizar a avaliação, marque "Extração concluída" para avançar. Atenção: ao concluir, a extração será ocultada e não poderá mais ser alterada. As três podem receber notas iguais ou diferentes — não há hierarquia implícita.'
    )

    # 2. Extracao 1 Intro
    content = content.replace(
        '''  <!-- ══════════════════════════════════════════
       EXTRAÇÃO I — sempre visível
       ══════════════════════════════════════════ -->
  <View className="extracao-bloco">''',
        '''  <!-- ══════════════════════════════════════════
       EXTRAÇÃO I — sempre visível
       ══════════════════════════════════════════ -->
  <View visibleWhen="choice-unselected" whenTagName="conclusao_col1" whenChoiceValue="Extração 1 concluída">
  <View className="extracao-bloco">'''
    )

    # 3. Extracao 1 Outro
    content = content.replace(
        '''            <Choice value="nao_consta_indev"  hint="'Não consta' onde a informação claramente existe"/>
          </Choices>
        </View>
      </View>
    </View>
  </View>

  <!-- ══════════════════════════════════════════
       EXTRAÇÃO II — aparece após avaliar I''',
        '''            <Choice value="nao_consta_indev"  hint="'Não consta' onde a informação claramente existe"/>
          </Choices>
        </View>
        <View className="extr-nota" style="margin-top: 15px;">
          <View className="label-sm"><Text name="l_conc1" value="Conclusão"/></View>
          <Choices name="conclusao_col1" toName="bloco1" choice="single">
            <Choice value="Extração 1 concluída" hint="Atenção: Ao concluir, esta extração será ocultada e não poderá mais ser alterada."/>
          </Choices>
        </View>
      </View>
    </View>
  </View>
  </View>

  <!-- ══════════════════════════════════════════
       EXTRAÇÃO II — aparece após avaliar I'''
    )

    # 4. Extracao 2 Intro
    content = content.replace(
        '''  <!-- ══════════════════════════════════════════
       EXTRAÇÃO II — aparece após avaliar I
       ══════════════════════════════════════════ -->
  <View className="progresso"
        visibleWhen="choice-selected"
        whenTagName="nota_col1"
        whenChoiceValue="4 - Excelente,3 - Adequado,2 - Ruim,1 - Inaceitável">
    <Text name="prog2" value="✓ Extração I avaliada — prossiga para a Extração II"/>
  </View>

  <View className="extracao-bloco"
        visibleWhen="choice-selected"
        whenTagName="nota_col1"
        whenChoiceValue="4 - Excelente,3 - Adequado,2 - Ruim,1 - Inaceitável">''',
        '''  <!-- ══════════════════════════════════════════
       EXTRAÇÃO II — aparece após avaliar I
       ══════════════════════════════════════════ -->
  <View visibleWhen="choice-selected" whenTagName="conclusao_col1" whenChoiceValue="Extração 1 concluída">
  <View visibleWhen="choice-unselected" whenTagName="conclusao_col2" whenChoiceValue="Extração 2 concluída">

  <View className="progresso">
    <Text name="prog2" value="✓ Extração I concluída — prossiga para a Extração II"/>
  </View>

  <View className="extracao-bloco">'''
    )

    # 5. Extracao 2 Outro
    content = content.replace(
        '''            <Choice value="nao_consta_indev"  hint="'Não consta' onde a informação claramente existe"/>
          </Choices>
        </View>
      </View>
    </View>
  </View>

  <!-- ══════════════════════════════════════════
       EXTRAÇÃO III — aparece após avaliar II''',
        '''            <Choice value="nao_consta_indev"  hint="'Não consta' onde a informação claramente existe"/>
          </Choices>
        </View>
        <View className="extr-nota" style="margin-top: 15px;">
          <View className="label-sm"><Text name="l_conc2" value="Conclusão"/></View>
          <Choices name="conclusao_col2" toName="bloco2" choice="single">
            <Choice value="Extração 2 concluída" hint="Atenção: Ao concluir, esta extração será ocultada e não poderá mais ser alterada."/>
          </Choices>
        </View>
      </View>
    </View>
  </View>
  </View>
  </View>

  <!-- ══════════════════════════════════════════
       EXTRAÇÃO III — aparece após avaliar II'''
    )

    # 6. Extracao 3 Intro
    content = content.replace(
        '''  <!-- ══════════════════════════════════════════
       EXTRAÇÃO III — aparece após avaliar II
       ══════════════════════════════════════════ -->
  <View className="progresso"
        visibleWhen="choice-selected"
        whenTagName="nota_col2"
        whenChoiceValue="4 - Excelente,3 - Adequado,2 - Ruim,1 - Inaceitável">
    <Text name="prog3" value="✓ Extração II avaliada — prossiga para a Extração III"/>
  </View>

  <View className="extracao-bloco"
        visibleWhen="choice-selected"
        whenTagName="nota_col2"
        whenChoiceValue="4 - Excelente,3 - Adequado,2 - Ruim,1 - Inaceitável">''',
        '''  <!-- ══════════════════════════════════════════
       EXTRAÇÃO III — aparece após avaliar II
       ══════════════════════════════════════════ -->
  <View visibleWhen="choice-selected" whenTagName="conclusao_col2" whenChoiceValue="Extração 2 concluída">
  <View visibleWhen="choice-unselected" whenTagName="conclusao_col3" whenChoiceValue="Extração 3 concluída">

  <View className="progresso">
    <Text name="prog3" value="✓ Extração II concluída — prossiga para a Extração III"/>
  </View>

  <View className="extracao-bloco">'''
    )

    # 7. Extracao 3 Outro
    content = content.replace(
        '''            <Choice value="nao_consta_indev"  hint="'Não consta' onde a informação claramente existe"/>
          </Choices>
        </View>
      </View>
    </View>
  </View>

  <!-- ══════════════════════════════════════════
       INDICADOR FINAL''',
        '''            <Choice value="nao_consta_indev"  hint="'Não consta' onde a informação claramente existe"/>
          </Choices>
        </View>
        <View className="extr-nota" style="margin-top: 15px;">
          <View className="label-sm"><Text name="l_conc3" value="Conclusão"/></View>
          <Choices name="conclusao_col3" toName="bloco3" choice="single">
            <Choice value="Extração 3 concluída" hint="Atenção: Ao concluir, esta extração será ocultada e não poderá mais ser alterada."/>
          </Choices>
        </View>
      </View>
    </View>
  </View>
  </View>
  </View>

  <!-- ══════════════════════════════════════════
       INDICADOR FINAL'''
    )

    # 8. Indicador Final
    content = content.replace(
        '''  <!-- ══════════════════════════════════════════
       INDICADOR FINAL
       ══════════════════════════════════════════ -->
  <View className="progresso"
        visibleWhen="choice-selected"
        whenTagName="nota_col3"
        whenChoiceValue="4 - Excelente,3 - Adequado,2 - Ruim,1 - Inaceitável">
    <Text name="prog_fim" value="✓ Todas as extrações avaliadas — pode submeter."/>
  </View>''',
        '''  <!-- ══════════════════════════════════════════
       INDICADOR FINAL
       ══════════════════════════════════════════ -->
  <View className="progresso"
        visibleWhen="choice-selected"
        whenTagName="conclusao_col3"
        whenChoiceValue="Extração 3 concluída">
    <Text name="prog_fim" value="✓ Todas as extrações concluídas — pode submeter."/>
  </View>'''
    )

    with open('template_label_studio_sequencial.xml', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    main()
