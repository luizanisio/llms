
"""
Módulo dedicado ao gerenciamento de templates de chat para treinamento com Unsloth.
Centraliza a lógica de aplicação de templates, formatação de datasets e configuração
de treinamento em apenas respostas (train_on_responses_only).
"""

import os
from typing import Any, Dict, List, Optional, Union
import torch
from unsloth.chat_templates import get_chat_template, CHAT_TEMPLATES, train_on_responses_only
from copy import deepcopy

class TreinarChatTemplate:
    """
    Gerencia a aplicação de templates de chat (Chat Templates) para modelos LLM.
    
    Esta classe centraliza:
    1. A escolha e configuração do template correto para o tokenizer
    2. A formatação de datasets para o formato de chat
    3. A configuração de treinamento focado apenas nas respostas (completions)
    """
    
    def __init__(self, tokenizer, model_name: str):
        """
        Inicializa o gerenciador de templates.
        
        Args:
            tokenizer: O tokenizer do modelo
            model_name: Nome do modelo base (usado para detecção automática de configs)
        """
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.template_name = None
        
        # Configura o template automaticamente na inicialização
        self._setup_chat_template()

    def _setup_chat_template(self):
        """
        Configura o chat template adequado para o modelo no tokenizer.
        """
        # Detecção baseada no nome do modelo
        _nm_teste = self.model_name.replace('-','').lower()
        key = "chatml" # fallback default
        
        # Lógica de seleção (similar a anterior, mas refinada)
        if 'gemma3' in _nm_teste: key = 'gemma' 
        elif 'gemma' in _nm_teste: key = 'gemma'
        elif 'qwen2.5' in _nm_teste: key = 'qwen-2.5'
        elif 'qwen2' in _nm_teste: key = 'qwen-2.5'
        elif 'qwen' in _nm_teste: key = 'chatml'
        elif 'llama3.3' in _nm_teste:  key = 'llama-3.3'
        elif 'llama3.2' in _nm_teste:  key = 'llama-3.2'
        elif 'llama3.1' in _nm_teste:  key = 'llama-3.1'
        elif 'llama3' in _nm_teste:  key = 'llama-3'
        elif 'llama' in _nm_teste:  key = 'llama'
        elif 'phi3' in _nm_teste: key = 'phi-3'
        elif 'mistral' in _nm_teste: key = 'mistral'
        
        if key not in CHAT_TEMPLATES:
            # Se não conhecemos o modelo explicitamente e ele JÁ tem template, mantemos
            if getattr(self.tokenizer, "chat_template", None):
                self.template_name = "custom/existing"
                print(f"ℹ️ Mantendo chat template existente do tokenizer (modelo não mapeado explicitamente para Unsloth)")
                return
            key = "chatml" 
            
        self.template_name = key
        
        try:
            # Forçamos a aplicação para garantir tokens especiais (EOS, BOS, UNK, PAD) corretos via Unsloth
            self.tokenizer = get_chat_template(self.tokenizer, chat_template=key)
            print(f"🔄 Chat Template aplicado: {key} (baseado em '{self.model_name}')")
        except Exception as e:
            print(f"⚠️ Aviso: Falha ao aplicar template '{key}': {e}")
            if not getattr(self.tokenizer, "chat_template", None):
                print("   Tentando 'chatml' genérico...")
                try:
                    self.tokenizer = get_chat_template(self.tokenizer, chat_template="chatml")
                    self.template_name = "chatml"
                except Exception as e2:
                    print(f"⚠️ Erro crítico: Falha ao aplicar chatml: {e2}")

    def verificar_dataset_formatado(self, dataset, n_samples=3):
        """
        Verifica se o dataset está corretamente formatado e tokenizável.
        Imprime exemplos e estatísticas.
        """
        print(f"\n🔍 Verificação de Dataset Formatado (n={n_samples}):")
        if "text" not in dataset.column_names:
            print("❌ Coluna 'text' não encontrada no dataset!")
            return
            
        for i in range(min(n_samples, len(dataset))):
            texto = dataset[i]["text"]
            print(f"  📝 Exemplo {i+1} (Length: {len(texto)} chars):")
            print(f"     Preview: {texto[:200]!r} ... {texto[-100:]!r}")
            
            # Teste de tokenização
            tokens = self.tokenizer(texto)
            input_ids = tokens["input_ids"]
            print(f"     Tokens: {len(input_ids)}")
            if len(input_ids) < 10:
                print(f"     ⚠️ Poucos tokens! Input IDs: {input_ids}")
            
            # Verifica integridade básica
            if not input_ids:
                print(f"     ❌ Tokenização gerou lista vazia!")
        
        print(f"✅ Verificação concluída.\n")

    def aplicar_train_on_responses_only(self, trainer) -> Any:
        """
        Aplica a configuração para treinar apenas nas respostas do assistente.
        
        Args:
            trainer: O objeto SFTTrainer instanciado
            
        Returns:
            O trainer configurado (ou o original se houver erro)
        """
        try:
            # Detecta as tags de instrução e resposta baseado no modelo/template
            _nm_teste = self.model_name.replace('-','').lower()
            
            instruction_part = None
            response_part = None
            
            # Tenta inferir pelos templates conhecidos
            if 'gemma' in _nm_teste:
                instruction_part = "<start_of_turn>user\n"
                response_part = "<start_of_turn>model\n"
            elif 'qwen' in _nm_teste:
                if self.template_name == 'qwen-2.5':
                    instruction_part = "<|im_start|>user\n"
                    response_part = "<|im_start|>assistant\n"
                else:
                    # ChatML padrão
                    instruction_part = "<|im_start|>user\n"
                    response_part = "<|im_start|>assistant\n"
            elif 'llama3' in _nm_teste:
                instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
                response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            elif 'deepseek' in _nm_teste:
                instruction_part = "<|im_start|>user\n"
                response_part = "<|im_start|>assistant\n"
            elif 'phi-3' in _nm_teste or self.template_name == 'phi-3':
                instruction_part = "<|user|>\n"
                response_part = "<|assistant|>\n"
            
            # Fallback para ChatML se não detectado
            if not instruction_part:
                instruction_part = "<|im_start|>user\n"
                response_part = "<|im_start|>assistant\n"
            
            # Aplica train_on_responses_only do unsloth
            trainer = train_on_responses_only(
                trainer,
                instruction_part=instruction_part,
                response_part=response_part,
            )
            print(f'   ✅ train_on_responses_only aplicado')
            print(f'      Tag User: "{instruction_part.strip()}"')
            print(f'      Tag Asst: "{response_part.strip()}"')
            return trainer
            
        except Exception as e:
            print(f'   ⚠️ Erro ao aplicar train_on_responses_only: {e}')
            print(f'   Continuando com treinamento padrão (full sequence loss)...')
            return trainer

    def formatar_dataset_coluna_text(self, dataset, num_proc: int = 2):
        """
        Formata o dataset adicionando a coluna 'text' com o chat template aplicado.
        Essa coluna é necessária para o SFTTrainer com dataset_text_field='text'.
        """
        if "text" in dataset.column_names:
            return dataset

        def _format_function(examples):
            texts = []
            # Verifica formato dos dados (batch)
            if "messages" in examples:
                # Lista de listas de mensagens
                for msgs in examples["messages"]:
                    texts.append(self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
            elif "prompt" in examples and "completion" in examples:
                # Formato prompt/completion
                for p, c in zip(examples["prompt"], examples["completion"]):
                    msgs = [{"role": "user", "content": str(p)}, {"role": "assistant", "content": str(c)}]
                    texts.append(self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
            else:
                # Fallback genérico ou erro silencioso (retorna string vazia para não quebrar map)
                primeira_col = list(examples.keys())[0]
                n_rows = len(examples[primeira_col])
                texts = [""] * n_rows
                
            return {"text": texts}

        print(f"🔄 Formatando dataset com chat template (proc={num_proc})...")
        try:
            return dataset.map(
                _format_function,
                batched=True,
                num_proc=num_proc,
                load_from_cache_file=False,
                desc="Aplicando Chat Template"
            )
        except Exception as e:
            print(f"❌ Erro ao formatar dataset: {e}")
            raise e

    def debug_template(self):
        """
        Imprime informações de debug sobre o template configurado.
        """
        print("\n🔍 DEBUG CHAT TEMPLATE:")
        print(f"  - Configurado: {self.template_name}")
        print(f"  - EOS Token: {self.tokenizer.eos_token} (id: {self.tokenizer.eos_token_id})")
        print(f"  - Pad Token: {self.tokenizer.pad_token} (id: {self.tokenizer.pad_token_id})")
        
        # Teste de formatação
        msgs = [
            {"role": "user", "content": "Olá, como vai?"},
            {"role": "assistant", "content": "Vou bem, obrigado!"}
        ]
        try:
            formated = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            print(f"  - Exemplo de formatação:\n{repr(formated)}")
        except Exception as e:
            print(f"  - ❌ Erro ao testar template: {e}") 
