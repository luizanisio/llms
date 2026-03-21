"""
Autor: Luiz Anísio
Fonte: https://github.com/luizanisio/llms/tree/main/src

Módulo de templates de chat usando HuggingFace Transformers nativo.
Substitui a dependência do unsloth.chat_templates, mantendo funcionalidade similar.

Funcionalidades:
- Aplicação de chat templates nativos
- Auto-detecção de template baseada no modelo
- Suporte a treinamento apenas em respostas (response_template masking)
"""

from typing import Optional
from transformers import PreTrainedTokenizer

# Mapeamento de modelos para templates conhecidos
MODEL_TEMPLATE_MAP = {
    'gemma': 'gemma',
    'gemma3': 'gemma',
    'qwen2.5': 'qwen2.5',
    'qwen2': 'qwen2.5',
    'qwen': 'chatml',
    'llama3.3': 'llama3',
    'llama3.2': 'llama3',
    'llama3.1': 'llama3',
    'llama3': 'llama3',
    'llama': 'llama2',
    'phi3': 'phi3',
    'mistral': 'mistral',
    'deepseek': 'deepseek',
}


class TreinarChatTemplate:
    """Gerencia templates de chat para modelos LLM usando HF Transformers.

    Esta classe:
    1. Auto-detecta e aplica o chat template correto
    2. Configura tokens especiais (pad, eos, bos)
    3. Suporta treinamento focado apenas nas respostas
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, model_name: str):
        """Inicializa o gerenciador de templates.

        Args:
            tokenizer: Tokenizer do modelo
            model_name: Nome do modelo base (para auto-detecção)
        """
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.template_name = None

        # Configura template automaticamente
        self._setup_chat_template()

    def _setup_chat_template(self):
        """Configura o chat template adequado para o modelo."""
        # Auto-detecta template baseado no nome do modelo
        model_lower = self.model_name.replace('-', '').lower()

        detected_template = None
        for model_key, template_name in MODEL_TEMPLATE_MAP.items():
            if model_key in model_lower:
                detected_template = template_name
                break

        # Se o tokenizer já tem um chat_template, mantém (a menos que seja genérico)
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # Verifica se é um template genérico/padrão (contém apenas {{ messages }})
            is_generic = (
                len(self.tokenizer.chat_template) < 100 and
                '{{ messages }}' in self.tokenizer.chat_template
            )

            if not is_generic:
                self.template_name = "native"
                print(f"ℹ️ Usando chat template nativo do tokenizer (modelo: {self.model_name})")
                self._ensure_special_tokens()
                return

        # Aplica template detectado (se houver)
        if detected_template:
            self.template_name = detected_template
            print(f"🔄 Chat Template detectado: {detected_template} (baseado em '{self.model_name}')")
        else:
            self.template_name = "default"
            print(f"⚠️ Template não reconhecido para '{self.model_name}'. Usando template nativo do tokenizer.")

        # Garante tokens especiais configurados
        self._ensure_special_tokens()

    def _ensure_special_tokens(self):
        """Garante que tokens especiais estão configurados corretamente."""
        # pad_token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                print(f"  ℹ️ pad_token configurado como eos_token: '{self.tokenizer.pad_token}'")
            else:
                # Fallback: adiciona token especial
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print(f"  ⚠️ pad_token adicionado como token especial: '[PAD]'")

        # eos_token
        if self.tokenizer.eos_token is None:
            print(f"  ⚠️ Aviso: eos_token não definido no tokenizer!")

        # bos_token
        if self.tokenizer.bos_token is None:
            print(f"  ℹ️ bos_token não definido (normal para alguns modelos)")

    def verificar_dataset_formatado(self, dataset, n_samples: int = 3):
        """Verifica se o dataset está corretamente formatado.

        Args:
            dataset: Dataset a verificar (deve ter coluna 'text')
            n_samples: Número de exemplos a imprimir
        """
        print(f"\n🔍 Verificação de Dataset Formatado (n={n_samples}):")

        if "text" not in dataset.column_names:
            print("❌ Coluna 'text' não encontrada no dataset!")
            print(f"   Colunas disponíveis: {dataset.column_names}")
            return

        for i in range(min(n_samples, len(dataset))):
            texto = dataset[i]["text"]
            print(f"\n  📝 Exemplo {i+1}:")
            print(f"     Length: {len(texto)} chars")

            # Tokeniza para verificar
            try:
                tokens = self.tokenizer(texto, truncation=True, max_length=512)
                num_tokens = len(tokens['input_ids'])
                print(f"     Tokens: {num_tokens}")

                # Mostra preview
                preview = texto[:200] + "..." if len(texto) > 200 else texto
                print(f"     Preview: {preview}")
            except Exception as e:
                print(f"     ⚠️ Erro ao tokenizar: {e}")

    def get_response_template(self) -> Optional[str]:
        """Retorna o template de resposta para masking (train on responses only).

        Retorna uma string que marca o início das respostas do assistente.
        Usado para treinar apenas nas respostas, ignorando prompts do usuário.

        Returns:
            String de template de resposta ou None se não aplicável
        """
        # Mapeamento de templates para seus marcadores de resposta
        response_templates = {
            'gemma': '<start_of_turn>model\n',
            'qwen2.5': '<|im_start|>assistant\n',
            'chatml': '<|im_start|>assistant\n',
            'llama3': '<|start_header_id|>assistant<|end_header_id|>\n\n',
            'llama2': '[/INST]',
            'phi3': '<|assistant|>\n',
            'mistral': '[/INST]',
            'deepseek': 'Assistant:',
        }

        return response_templates.get(self.template_name)

    def formatar_dataset_coluna_text(self, dataset, num_proc: int = 2):
        """Formata dataset adicionando coluna 'text' com chat template aplicado.

        Esta coluna é necessária para SFTTrainer com dataset_text_field='text'.

        Args:
            dataset: Dataset HuggingFace
            num_proc: Número de processos para paralelização

        Returns:
            Dataset com coluna 'text' adicionada
        """
        if "text" in dataset.column_names:
            return dataset

        def _format_function(examples):
            texts = []
            # Verifica formato dos dados (batch)
            if "messages" in examples:
                # Lista de listas de mensagens (formato chat)
                for msgs in examples["messages"]:
                    texts.append(
                        self.tokenizer.apply_chat_template(
                            msgs,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                    )
            elif "prompt" in examples and "completion" in examples:
                # Formato prompt/completion
                for p, c in zip(examples["prompt"], examples["completion"]):
                    msgs = [
                        {"role": "user", "content": str(p)},
                        {"role": "assistant", "content": str(c)}
                    ]
                    texts.append(
                        self.tokenizer.apply_chat_template(
                            msgs,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                    )
            else:
                # Fallback: gera strings vazias para não quebrar map
                primeira_col = list(examples.keys())[0]
                n_rows = len(examples[primeira_col])
                texts = [""] * n_rows

            return {"text": texts}

        # Garante que num_proc não exceda número de registros
        safe_proc = min(num_proc, len(dataset)) if len(dataset) > 0 else 1
        print(f"🔄 Formatando dataset com chat template (proc={safe_proc})...")

        try:
            return dataset.map(
                _format_function,
                batched=True,
                num_proc=safe_proc,
                load_from_cache_file=False,
                desc="Aplicando Chat Template"
            )
        except Exception as e:
            print(f"❌ Erro ao formatar dataset: {e}")
            raise

    def aplicar_train_on_responses_only(self, trainer):
        """Configura trainer para treinar apenas nas respostas do assistente.

        Usa DataCollatorForCompletionOnlyLM do trl para mascarar prompts.

        Args:
            trainer: SFTTrainer instanciado

        Returns:
            Trainer configurado (ou original se houver erro)
        """
        try:
            # Obtém template de resposta
            response_template = self.get_response_template()

            if not response_template:
                print(f"⚠️  Template de resposta não definido para '{self.template_name}'")
                print(f"   Treinando em sequência completa (sem masking)...")
                return trainer

            # Cria data collator para completion-only
            from trl import DataCollatorForCompletionOnlyLM

            # Salva estado original (caso precise restaurar)
            collator_original = trainer.data_collator

            try:
                # Aplica novo collator
                trainer.data_collator = DataCollatorForCompletionOnlyLM(
                    response_template=response_template,
                    tokenizer=self.tokenizer,
                    mlm=False,
                )

                print(f"✅ train_on_responses_only aplicado")
                print(f"   Response template: \"{response_template.strip()}\"")

                return trainer

            except Exception as e:
                print(f"⚠️  Erro ao configurar completion-only collator: {e}")
                print(f"   Restaurando collator original...")
                trainer.data_collator = collator_original
                return trainer

        except Exception as e:
            print(f"⚠️  Erro ao aplicar train_on_responses_only: {e}")
            print(f"   Continuando com treinamento padrão (full sequence loss)...")
            return trainer

    def apply_chat_template(self, messages: list, add_generation_prompt: bool = False) -> str:
        """Aplica o chat template a uma lista de mensagens.

        Args:
            messages: Lista de mensagens no formato [{"role": "user", "content": "..."}]
            add_generation_prompt: Se True, adiciona prompt para geração de resposta

        Returns:
            String formatada com o template de chat
        """
        if not hasattr(self.tokenizer, 'apply_chat_template'):
            raise NotImplementedError(
                f"Tokenizer {type(self.tokenizer).__name__} não suporta apply_chat_template"
            )

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def info(self) -> dict:
        """Retorna informações sobre a configuração atual."""
        return {
            "model_name": self.model_name,
            "template_name": self.template_name,
            "pad_token": self.tokenizer.pad_token,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token": self.tokenizer.eos_token,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token": self.tokenizer.bos_token,
            "bos_token_id": self.tokenizer.bos_token_id,
            "has_chat_template": hasattr(self.tokenizer, 'chat_template'),
        }


def get_data_collator_for_completion_only(
    tokenizer: PreTrainedTokenizer,
    response_template: str,
    instruction_template: Optional[str] = None,
    mlm: bool = False,
):
    """Cria um data collator que treina apenas nas respostas (completions).

    Args:
        tokenizer: Tokenizer do modelo
        response_template: String que marca início das respostas (ex: "<|assistant|>")
        instruction_template: String que marca início das instruções (opcional)
        mlm: Se True, usa masked language modeling (False para causal LM)

    Returns:
        DataCollatorForCompletionOnlyLM do trl
    """
    from trl import DataCollatorForCompletionOnlyLM

    return DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        instruction_template=instruction_template,
        tokenizer=tokenizer,
        mlm=mlm,
    )
