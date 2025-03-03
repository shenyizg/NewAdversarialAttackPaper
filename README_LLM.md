# Latest Large Language Model Attack Papers
**update at 2025-03-03 09:51:19**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference**

cs.AI

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2407.00075v4) [paper-pdf](http://arxiv.org/pdf/2407.00075v4)

**Authors**: Anton Xue, Avishree Khare, Rajeev Alur, Surbhi Goel, Eric Wong

**Abstract**: We study how to subvert large language models (LLMs) from following prompt-specified rules. We first formalize rule-following as inference in propositional Horn logic, a mathematical system in which rules have the form "if $P$ and $Q$, then $R$" for some propositions $P$, $Q$, and $R$. Next, we prove that although small transformers can faithfully follow such rules, maliciously crafted prompts can still mislead both theoretical constructions and models learned from data. Furthermore, we demonstrate that popular attack algorithms on LLMs find adversarial prompts and induce attention patterns that align with our theory. Our novel logic-based framework provides a foundation for studying LLMs in rule-based settings, enabling a formal analysis of tasks like logical reasoning and jailbreak attacks.



## **2. Beyond Natural Language Perplexity: Detecting Dead Code Poisoning in Code Generation Datasets**

cs.CL

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2502.20246v2) [paper-pdf](http://arxiv.org/pdf/2502.20246v2)

**Authors**: Chi-Chien Tsai, Chia-Mu Yu, Ying-Dar Lin, Yu-Sung Wu, Wei-Bin Lee

**Abstract**: The increasing adoption of large language models (LLMs) for code-related tasks has raised concerns about the security of their training datasets. One critical threat is dead code poisoning, where syntactically valid but functionally redundant code is injected into training data to manipulate model behavior. Such attacks can degrade the performance of neural code search systems, leading to biased or insecure code suggestions. Existing detection methods, such as token-level perplexity analysis, fail to effectively identify dead code due to the structural and contextual characteristics of programming languages. In this paper, we propose DePA (Dead Code Perplexity Analysis), a novel line-level detection and cleansing method tailored to the structural properties of code. DePA computes line-level perplexity by leveraging the contextual relationships between code lines and identifies anomalous lines by comparing their perplexity to the overall distribution within the file. Our experiments on benchmark datasets demonstrate that DePA significantly outperforms existing methods, achieving 0.14-0.19 improvement in detection F1-score and a 44-65% increase in poisoned segment localization precision. Furthermore, DePA enhances detection speed by 0.62-23x, making it practical for large-scale dataset cleansing. Overall, by addressing the unique challenges of dead code poisoning, DePA provides a robust and efficient solution for safeguarding the integrity of code generation model training datasets.



## **3. Behind the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models**

cs.CR

12 pages. 6 figures

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2502.19883v2) [paper-pdf](http://arxiv.org/pdf/2502.19883v2)

**Authors**: Sibo Yi, Tianshuo Cong, Xinlei He, Qi Li, Jiaxing Song

**Abstract**: Small language models (SLMs) have become increasingly prominent in the deployment on edge devices due to their high efficiency and low computational cost. While researchers continue to advance the capabilities of SLMs through innovative training strategies and model compression techniques, the security risks of SLMs have received considerably less attention compared to large language models (LLMs).To fill this gap, we provide a comprehensive empirical study to evaluate the security performance of 13 state-of-the-art SLMs under various jailbreak attacks. Our experiments demonstrate that most SLMs are quite susceptible to existing jailbreak attacks, while some of them are even vulnerable to direct harmful prompts.To address the safety concerns, we evaluate several representative defense methods and demonstrate their effectiveness in enhancing the security of SLMs. We further analyze the potential security degradation caused by different SLM techniques including architecture compression, quantization, knowledge distillation, and so on. We expect that our research can highlight the security challenges of SLMs and provide valuable insights to future work in developing more robust and secure SLMs.



## **4. Trustworthy AI-Generative Content for Intelligent Network Service: Robustness, Security, and Fairness**

cs.CR

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2405.05930v2) [paper-pdf](http://arxiv.org/pdf/2405.05930v2)

**Authors**: Siyuan Li, Xi Lin, Yaju Liu, Xiang Chen, Jianhua Li

**Abstract**: AI-generated content (AIGC) models, represented by large language models (LLM), have revolutionized content creation. High-speed next-generation communication technology is an ideal platform for providing powerful AIGC network services. At the same time, advanced AIGC techniques can also make future network services more intelligent, especially various online content generation services. However, the significant untrustworthiness concerns of current AIGC models, such as robustness, security, and fairness, greatly affect the credibility of intelligent network services, especially in ensuring secure AIGC services. This paper proposes TrustGAIN, a trustworthy AIGC framework that incorporates robust, secure, and fair network services. We first discuss the robustness to adversarial attacks faced by AIGC models in network systems and the corresponding protection issues. Subsequently, we emphasize the importance of avoiding unsafe and illegal services and ensuring the fairness of the AIGC network services. Then as a case study, we propose a novel sentiment analysis-based detection method to guide the robust detection of unsafe content in network services. We conduct our experiments on fake news, malicious code, and unsafe review datasets to represent LLM application scenarios. Our results indicate that TrustGAIN is an exploration of future networks that can support trustworthy AIGC network services.



## **5. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

cs.CY

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.12659v3) [paper-pdf](http://arxiv.org/pdf/2502.12659v3)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models, such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source R1 models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on R1 is needed. (2) The distilled reasoning model shows poorer safety performance compared to its safety-aligned base models. (3) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (4) The thinking process in R1 models pose greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.



## **6. Foot-In-The-Door: A Multi-turn Jailbreak for LLMs**

cs.CL

19 pages, 8 figures

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2502.19820v2) [paper-pdf](http://arxiv.org/pdf/2502.19820v2)

**Authors**: Zixuan Weng, Xiaolong Jin, Jinyuan Jia, Xiangyu Zhang

**Abstract**: Ensuring AI safety is crucial as large language models become increasingly integrated into real-world applications. A key challenge is jailbreak, where adversarial prompts bypass built-in safeguards to elicit harmful disallowed outputs. Inspired by psychological foot-in-the-door principles, we introduce FITD,a novel multi-turn jailbreak method that leverages the phenomenon where minor initial commitments lower resistance to more significant or more unethical transgressions. Our approach progressively escalates the malicious intent of user queries through intermediate bridge prompts and aligns the model's response by itself to induce toxic responses. Extensive experimental results on two jailbreak benchmarks demonstrate that FITD achieves an average attack success rate of 94% across seven widely used models, outperforming existing state-of-the-art methods. Additionally, we provide an in-depth analysis of LLM self-corruption, highlighting vulnerabilities in current alignment strategies and emphasizing the risks inherent in multi-turn interactions. The code is available at https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak.



## **7. Tokens for Learning, Tokens for Unlearning: Mitigating Membership Inference Attacks in Large Language Models via Dual-Purpose Training**

cs.LG

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19726v1) [paper-pdf](http://arxiv.org/pdf/2502.19726v1)

**Authors**: Toan Tran, Ruixuan Liu, Li Xiong

**Abstract**: Large language models (LLMs) have become the backbone of modern natural language processing but pose privacy concerns about leaking sensitive training data. Membership inference attacks (MIAs), which aim to infer whether a sample is included in a model's training dataset, can serve as a foundation for broader privacy threats. Existing defenses designed for traditional classification models do not account for the sequential nature of text data. As a result, they either require significant computational resources or fail to effectively mitigate privacy risks in LLMs. In this work, we propose a lightweight yet effective empirical privacy defense for protecting training data of language modeling by leveraging the token-specific characteristics. By analyzing token dynamics during training, we propose a token selection strategy that categorizes tokens into hard tokens for learning and memorized tokens for unlearning. Subsequently, our training-phase defense optimizes a novel dual-purpose token-level loss to achieve a Pareto-optimal balance between utility and privacy. Extensive experiments demonstrate that our approach not only provides strong protection against MIAs but also improves language modeling performance by around 10\% across various LLM architectures and datasets compared to the baselines.



## **8. Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models**

cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.18290v2) [paper-pdf](http://arxiv.org/pdf/2502.18290v2)

**Authors**: Zhaoyi Liu, Huan Zhang

**Abstract**: Self-supervised learning (SSL) vision encoders learn high-quality image representations and thus have become a vital part of developing vision modality of large vision language models (LVLMs). Due to the high cost of training such encoders, pre-trained encoders are widely shared and deployed into many LVLMs, which are security-critical or bear societal significance. Under this practical scenario, we reveal a new backdoor threat that significant visual hallucinations can be induced into these LVLMs by merely compromising vision encoders. Because of the sharing and reuse of these encoders, many downstream LVLMs may inherit backdoor behaviors from encoders, leading to widespread backdoors. In this work, we propose BadVision, the first method to exploit this vulnerability in SSL vision encoders for LVLMs with novel trigger optimization and backdoor learning techniques. We evaluate BadVision on two types of SSL encoders and LVLMs across eight benchmarks. We show that BadVision effectively drives the LVLMs to attacker-chosen hallucination with over 99% attack success rate, causing a 77.6% relative visual understanding error while maintaining the stealthiness. SoTA backdoor detection methods cannot detect our attack effectively.



## **9. MAA: Meticulous Adversarial Attack against Vision-Language Pre-trained Models**

cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.08079v2) [paper-pdf](http://arxiv.org/pdf/2502.08079v2)

**Authors**: Peng-Fei Zhang, Guangdong Bai, Zi Huang

**Abstract**: Current adversarial attacks for evaluating the robustness of vision-language pre-trained (VLP) models in multi-modal tasks suffer from limited transferability, where attacks crafted for a specific model often struggle to generalize effectively across different models, limiting their utility in assessing robustness more broadly. This is mainly attributed to the over-reliance on model-specific features and regions, particularly in the image modality. In this paper, we propose an elegant yet highly effective method termed Meticulous Adversarial Attack (MAA) to fully exploit model-independent characteristics and vulnerabilities of individual samples, achieving enhanced generalizability and reduced model dependence. MAA emphasizes fine-grained optimization of adversarial images by developing a novel resizing and sliding crop (RScrop) technique, incorporating a multi-granularity similarity disruption (MGSD) strategy. Extensive experiments across diverse VLP models, multiple benchmark datasets, and a variety of downstream tasks demonstrate that MAA significantly enhances the effectiveness and transferability of adversarial attacks. A large cohort of performance studies is conducted to generate insights into the effectiveness of various model configurations, guiding future advancements in this domain.



## **10. M-LLM Based Video Frame Selection for Efficient Video Understanding**

cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19680v1) [paper-pdf](http://arxiv.org/pdf/2502.19680v1)

**Authors**: Kai Hu, Feng Gao, Xiaohan Nie, Peng Zhou, Son Tran, Tal Neiman, Lingyun Wang, Mubarak Shah, Raffay Hamid, Bing Yin, Trishul Chilimbi

**Abstract**: Recent advances in Multi-Modal Large Language Models (M-LLMs) show promising results in video reasoning. Popular Multi-Modal Large Language Model (M-LLM) frameworks usually apply naive uniform sampling to reduce the number of video frames that are fed into an M-LLM, particularly for long context videos. However, it could lose crucial context in certain periods of a video, so that the downstream M-LLM may not have sufficient visual information to answer a question. To attack this pain point, we propose a light-weight M-LLM -based frame selection method that adaptively select frames that are more relevant to users' queries. In order to train the proposed frame selector, we introduce two supervision signals (i) Spatial signal, where single frame importance score by prompting a M-LLM; (ii) Temporal signal, in which multiple frames selection by prompting Large Language Model (LLM) using the captions of all frame candidates. The selected frames are then digested by a frozen downstream video M-LLM for visual reasoning and question answering. Empirical results show that the proposed M-LLM video frame selector improves the performances various downstream video Large Language Model (video-LLM) across medium (ActivityNet, NExT-QA) and long (EgoSchema, LongVideoBench) context video question answering benchmarks.



## **11. Improving Adversarial Transferability in MLLMs via Dynamic Vision-Language Alignment Attack**

cs.CV

arXiv admin note: text overlap with arXiv:2403.09766

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19672v1) [paper-pdf](http://arxiv.org/pdf/2502.19672v1)

**Authors**: Chenhe Gu, Jindong Gu, Andong Hua, Yao Qin

**Abstract**: Multimodal Large Language Models (MLLMs), built upon LLMs, have recently gained attention for their capabilities in image recognition and understanding. However, while MLLMs are vulnerable to adversarial attacks, the transferability of these attacks across different models remains limited, especially under targeted attack setting. Existing methods primarily focus on vision-specific perturbations but struggle with the complex nature of vision-language modality alignment. In this work, we introduce the Dynamic Vision-Language Alignment (DynVLA) Attack, a novel approach that injects dynamic perturbations into the vision-language connector to enhance generalization across diverse vision-language alignment of different models. Our experimental results show that DynVLA significantly improves the transferability of adversarial examples across various MLLMs, including BLIP2, InstructBLIP, MiniGPT4, LLaVA, and closed-source models such as Gemini.



## **12. H-CoT: Hijacking the Chain-of-Thought Safety Reasoning Mechanism to Jailbreak Large Reasoning Models, Including OpenAI o1/o3, DeepSeek-R1, and Gemini 2.0 Flash Thinking**

cs.CL

Website: https://maliciouseducator.org/

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.12893v2) [paper-pdf](http://arxiv.org/pdf/2502.12893v2)

**Authors**: Martin Kuo, Jianyi Zhang, Aolin Ding, Qinsi Wang, Louis DiValentin, Yujia Bao, Wei Wei, Hai Li, Yiran Chen

**Abstract**: Large Reasoning Models (LRMs) have recently extended their powerful reasoning capabilities to safety checks-using chain-of-thought reasoning to decide whether a request should be answered. While this new approach offers a promising route for balancing model utility and safety, its robustness remains underexplored. To address this gap, we introduce Malicious-Educator, a benchmark that disguises extremely dangerous or malicious requests beneath seemingly legitimate educational prompts. Our experiments reveal severe security flaws in popular commercial-grade LRMs, including OpenAI o1/o3, DeepSeek-R1, and Gemini 2.0 Flash Thinking. For instance, although OpenAI's o1 model initially maintains a high refusal rate of about 98%, subsequent model updates significantly compromise its safety; and attackers can easily extract criminal strategies from DeepSeek-R1 and Gemini 2.0 Flash Thinking without any additional tricks. To further highlight these vulnerabilities, we propose Hijacking Chain-of-Thought (H-CoT), a universal and transferable attack method that leverages the model's own displayed intermediate reasoning to jailbreak its safety reasoning mechanism. Under H-CoT, refusal rates sharply decline-dropping from 98% to below 2%-and, in some instances, even transform initially cautious tones into ones that are willing to provide harmful content. We hope these findings underscore the urgent need for more robust safety mechanisms to preserve the benefits of advanced reasoning capabilities without compromising ethical standards.



## **13. Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack**

cs.CR

Accepted at USENIX Security 2025

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2404.01833v3) [paper-pdf](http://arxiv.org/pdf/2404.01833v3)

**Authors**: Mark Russinovich, Ahmed Salem, Ronen Eldan

**Abstract**: Large Language Models (LLMs) have risen significantly in popularity and are increasingly being adopted across multiple applications. These LLMs are heavily aligned to resist engaging in illegal or unethical topics as a means to avoid contributing to responsible AI harms. However, a recent line of attacks, known as jailbreaks, seek to overcome this alignment. Intuitively, jailbreak attacks aim to narrow the gap between what the model can do and what it is willing to do. In this paper, we introduce a novel jailbreak attack called Crescendo. Unlike existing jailbreak methods, Crescendo is a simple multi-turn jailbreak that interacts with the model in a seemingly benign manner. It begins with a general prompt or question about the task at hand and then gradually escalates the dialogue by referencing the model's replies progressively leading to a successful jailbreak. We evaluate Crescendo on various public systems, including ChatGPT, Gemini Pro, Gemini-Ultra, LlaMA-2 70b and LlaMA-3 70b Chat, and Anthropic Chat. Our results demonstrate the strong efficacy of Crescendo, with it achieving high attack success rates across all evaluated models and tasks. Furthermore, we present Crescendomation, a tool that automates the Crescendo attack and demonstrate its efficacy against state-of-the-art models through our evaluations. Crescendomation surpasses other state-of-the-art jailbreaking techniques on the AdvBench subset dataset, achieving 29-61% higher performance on GPT-4 and 49-71% on Gemini-Pro. Finally, we also demonstrate Crescendo's ability to jailbreak multimodal models.



## **14. Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs**

cs.CR

15 pages, 12 figures

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19041v1) [paper-pdf](http://arxiv.org/pdf/2502.19041v1)

**Authors**: Shiyu Xiang, Ansen Zhang, Yanfei Cao, Yang Fan, Ronghao Chen

**Abstract**: Although Aligned Large Language Models (LLMs) are trained to refuse harmful requests, they remain vulnerable to jailbreak attacks. Unfortunately, existing methods often focus on surface-level patterns, overlooking the deeper attack essences. As a result, defenses fail when attack prompts change, even though the underlying "attack essence" remains the same. To address this issue, we introduce EDDF, an \textbf{E}ssence-\textbf{D}riven \textbf{D}efense \textbf{F}ramework Against Jailbreak Attacks in LLMs. EDDF is a plug-and-play input-filtering method and operates in two stages: 1) offline essence database construction, and 2) online adversarial query detection. The key idea behind EDDF is to extract the "attack essence" from a diverse set of known attack instances and store it in an offline vector database. Experimental results demonstrate that EDDF significantly outperforms existing methods by reducing the Attack Success Rate by at least 20\%, underscoring its superior robustness against jailbreak attacks.



## **15. Towards Label-Only Membership Inference Attack against Pre-trained Large Language Models**

cs.CR

Accepted by USENIX Security 2025

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18943v1) [paper-pdf](http://arxiv.org/pdf/2502.18943v1)

**Authors**: Yu He, Boheng Li, Liu Liu, Zhongjie Ba, Wei Dong, Yiming Li, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Membership Inference Attacks (MIAs) aim to predict whether a data sample belongs to the model's training set or not. Although prior research has extensively explored MIAs in Large Language Models (LLMs), they typically require accessing to complete output logits (\ie, \textit{logits-based attacks}), which are usually not available in practice. In this paper, we study the vulnerability of pre-trained LLMs to MIAs in the \textit{label-only setting}, where the adversary can only access generated tokens (text). We first reveal that existing label-only MIAs have minor effects in attacking pre-trained LLMs, although they are highly effective in inferring fine-tuning datasets used for personalized LLMs. We find that their failure stems from two main reasons, including better generalization and overly coarse perturbation. Specifically, due to the extensive pre-training corpora and exposing each sample only a few times, LLMs exhibit minimal robustness differences between members and non-members. This makes token-level perturbations too coarse to capture such differences.   To alleviate these problems, we propose \textbf{PETAL}: a label-only membership inference attack based on \textbf{PE}r-\textbf{T}oken sem\textbf{A}ntic simi\textbf{L}arity. Specifically, PETAL leverages token-level semantic similarity to approximate output probabilities and subsequently calculate the perplexity. It finally exposes membership based on the common assumption that members are `better' memorized and have smaller perplexity. We conduct extensive experiments on the WikiMIA benchmark and the more challenging MIMIR benchmark. Empirically, our PETAL performs better than the extensions of existing label-only attacks against personalized LLMs and even on par with other advanced logit-based attacks across all metrics on five prevalent open-source LLMs.



## **16. JailBench: A Comprehensive Chinese Security Assessment Benchmark for Large Language Models**

cs.CL

12 pages, 5 figures, accepted at PAKDD 2025

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18935v1) [paper-pdf](http://arxiv.org/pdf/2502.18935v1)

**Authors**: Shuyi Liu, Simiao Cui, Haoran Bu, Yuming Shang, Xi Zhang

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various applications, highlighting the urgent need for comprehensive safety evaluations. In particular, the enhanced Chinese language proficiency of LLMs, combined with the unique characteristics and complexity of Chinese expressions, has driven the emergence of Chinese-specific benchmarks for safety assessment. However, these benchmarks generally fall short in effectively exposing LLM safety vulnerabilities. To address the gap, we introduce JailBench, the first comprehensive Chinese benchmark for evaluating deep-seated vulnerabilities in LLMs, featuring a refined hierarchical safety taxonomy tailored to the Chinese context. To improve generation efficiency, we employ a novel Automatic Jailbreak Prompt Engineer (AJPE) framework for JailBench construction, which incorporates jailbreak techniques to enhance assessing effectiveness and leverages LLMs to automatically scale up the dataset through context-learning. The proposed JailBench is extensively evaluated over 13 mainstream LLMs and achieves the highest attack success rate against ChatGPT compared to existing Chinese benchmarks, underscoring its efficacy in identifying latent vulnerabilities in LLMs, as well as illustrating the substantial room for improvement in the security and trustworthiness of LLMs within the Chinese context. Our benchmark is publicly available at https://github.com/STAIR-BUPT/JailBench.



## **17. Defense Against Prompt Injection Attack by Leveraging Attack Techniques**

cs.CR

9 pages

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2411.00459v3) [paper-pdf](http://arxiv.org/pdf/2411.00459v3)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song, Dekai Wu, Bryan Hooi

**Abstract**: With the advancement of technology, large language models (LLMs) have achieved remarkable performance across various natural language processing (NLP) tasks, powering LLM-integrated applications like Microsoft Copilot. However, as LLMs continue to evolve, new vulnerabilities, especially prompt injection attacks arise. These attacks trick LLMs into deviating from the original input instructions and executing the attacker's instructions injected in data content, such as retrieved results. Recent attack methods leverage LLMs' instruction-following abilities and their inabilities to distinguish instructions injected in the data content, and achieve a high attack success rate (ASR). When comparing the attack and defense methods, we interestingly find that they share similar design goals, of inducing the model to ignore unwanted instructions and instead to execute wanted instructions. Therefore, we raise an intuitive question: Could these attack techniques be utilized for defensive purposes? In this paper, we invert the intention of prompt injection methods to develop novel defense methods based on previous training-free attack methods, by repeating the attack process but with the original input instruction rather than the injected instruction. Our comprehensive experiments demonstrate that our defense techniques outperform existing training-free defense approaches, achieving state-of-the-art results.



## **18. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

cs.CL

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.01386v2) [paper-pdf](http://arxiv.org/pdf/2502.01386v2)

**Authors**: Yuyang Gong, Zhuo Chen, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.



## **19. CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification**

cs.CV

accepted by ICLR 2025

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.18176v1) [paper-pdf](http://arxiv.org/pdf/2502.18176v1)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: In this paper, we aim to build an adversarially robust zero-shot image classifier. We ground our work on CLIP, a vision-language pre-trained encoder model that can perform zero-shot classification by matching an image with text prompts ``a photo of a <class-name>.''. Purification is the path we choose since it does not require adversarial training on specific attack types and thus can cope with any foreseen attacks. We then formulate purification risk as the KL divergence between the joint distributions of the purification process of denoising the adversarial samples and the attack process of adding perturbations to benign samples, through bidirectional Stochastic Differential Equations (SDEs). The final derived results inspire us to explore purification in the multi-modal latent space of CLIP. We propose two variants for our CLIPure approach: CLIPure-Diff which models the likelihood of images' latent vectors with the DiffusionPrior module in DaLLE-2 (modeling the generation process of CLIP's latent vectors), and CLIPure-Cos which models the likelihood with the cosine similarity between the embeddings of an image and ``a photo of a.''. As far as we know, CLIPure is the first purification method in multi-modal latent space and CLIPure-Cos is the first purification method that is not based on generative models, which substantially improves defense efficiency. We conducted extensive experiments on CIFAR-10, ImageNet, and 13 datasets that previous CLIP-based defense methods used for evaluating zero-shot classification robustness. Results show that CLIPure boosts the SOTA robustness by a large margin, e.g., from 71.7% to 91.1% on CIFAR10, from 59.6% to 72.6% on ImageNet, and 108% relative improvements of average robustness on the 13 datasets over previous SOTA. The code is available at https://github.com/TMLResearchGroup-CAS/CLIPure.



## **20. Towards Robust and Secure Embodied AI: A Survey on Vulnerabilities and Attacks**

cs.CR

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.13175v2) [paper-pdf](http://arxiv.org/pdf/2502.13175v2)

**Authors**: Wenpeng Xing, Minghao Li, Mohan Li, Meng Han

**Abstract**: Embodied AI systems, including robots and autonomous vehicles, are increasingly integrated into real-world applications, where they encounter a range of vulnerabilities stemming from both environmental and system-level factors. These vulnerabilities manifest through sensor spoofing, adversarial attacks, and failures in task and motion planning, posing significant challenges to robustness and safety. Despite the growing body of research, existing reviews rarely focus specifically on the unique safety and security challenges of embodied AI systems. Most prior work either addresses general AI vulnerabilities or focuses on isolated aspects, lacking a dedicated and unified framework tailored to embodied AI. This survey fills this critical gap by: (1) categorizing vulnerabilities specific to embodied AI into exogenous (e.g., physical attacks, cybersecurity threats) and endogenous (e.g., sensor failures, software flaws) origins; (2) systematically analyzing adversarial attack paradigms unique to embodied AI, with a focus on their impact on perception, decision-making, and embodied interaction; (3) investigating attack vectors targeting large vision-language models (LVLMs) and large language models (LLMs) within embodied systems, such as jailbreak attacks and instruction misinterpretation; (4) evaluating robustness challenges in algorithms for embodied perception, decision-making, and task planning; and (5) proposing targeted strategies to enhance the safety and reliability of embodied AI systems. By integrating these dimensions, we provide a comprehensive framework for understanding the interplay between vulnerabilities and safety in embodied AI.



## **21. Efficient Safety Retrofitting Against Jailbreaking for LLMs**

cs.CL

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.13603v2) [paper-pdf](http://arxiv.org/pdf/2502.13603v2)

**Authors**: Dario Garcia-Gasulla, Adrian Tormos, Anna Arias-Duart, Daniel Hinjos, Oscar Molina-Sedano, Ashwin Kumar Gururajan, Maria Eugenia Cardello

**Abstract**: Direct Preference Optimization (DPO) is an efficient alignment technique that steers LLMs towards preferable outputs by training on preference data, bypassing the need for explicit reward models. Its simplicity enables easy adaptation to various domains and safety requirements. This paper examines DPO's effectiveness in model safety against jailbreaking attacks while minimizing data requirements and training costs. We introduce Egida, a dataset expanded from multiple sources, which includes 27 different safety topics and 18 different attack styles, complemented with synthetic and human labels. This data is used to boost the safety of state-of-the-art LLMs (Llama-3.1-8B/70B-Instruct, Qwen-2.5-7B/72B-Instruct) across topics and attack styles. In addition to safety evaluations, we assess their post-alignment performance degradation in general purpose tasks, and their tendency to over refusal. Following the proposed methodology, trained models reduce their Attack Success Rate by 10%-30%, using small training efforts (2,000 samples) with low computational cost (3\$ for 8B models, 20\$ for 72B models). Safety aligned models generalize to unseen topics and attack styles, with the most successful attack style reaching a success rate around 5%. Size and family are found to strongly influence model malleability towards safety, pointing at the importance of pre-training choices. To validate our findings, a large independent assessment of human preference agreement with Llama-Guard-3-8B is conducted by the authors and the associated dataset Egida-HSafe is released. Overall, this study illustrates how affordable and accessible it is to enhance LLM safety using DPO while outlining its current limitations. All datasets and models are released to enable reproducibility and further research.



## **22. S$^4$ST: A Strong, Self-transferable, faSt, and Simple Scale Transformation for Transferable Targeted Attack**

cs.CR

16 pages, 18 figures

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2410.13891v2) [paper-pdf](http://arxiv.org/pdf/2410.13891v2)

**Authors**: Yongxiang Liu, Bowen Peng, Li Liu, Xiang Li

**Abstract**: Transferable Targeted Attacks (TTAs), which aim to deceive black-box models into predicting specific erroneous labels, face significant challenges due to severe overfitting to surrogate models. Although modifying image features to generate robust semantic patterns of the target class is a promising approach, existing methods heavily rely on large-scale additional data. This dependence undermines the fair evaluation of TTA threats, potentially leading to a false sense of security or unnecessary overreactions. In this paper, we introduce two blind measures, surrogate self-alignment and self-transferability, to analyze the effectiveness and correlations of basic transformations, to enhance data-free attacks under strict black-box constraints. Our findings challenge conventional assumptions: (1) Attacking simple scaling transformations uniquely enhances targeted transferability, outperforming other basic transformations and rivaling leading complex methods; (2) Geometric and color transformations exhibit high internal redundancy despite weak inter-category correlations. These insights drive the design and tuning of S4ST (Strong, Self-transferable, faSt, Simple Scale Transformation), which integrates dimensionally consistent scaling, complementary low-redundancy transformations, and block-wise operations. Extensive experiments on the ImageNet-Compatible dataset demonstrate that S4ST achieves a 77.7% average targeted success rate (tSuc), surpassing existing transformations (+17.2% over H-Aug with only 26% computational time) and SOTA TTA solutions (+6.2% over SASD-WS with 1.2M samples for post-training). Notably, it attains 69.6% and 55.3% average tSuc against three commercial APIs and vision-language models, respectively. This work establishes a new SOTA for TTAs, highlights their potential threats, and calls for a reevaluation of the data dependency in achieving targeted transferability.



## **23. MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks**

cs.LG

Code is available at https://github.com/HyeonjeongHa/MM-PoisonRAG

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.17832v1) [paper-pdf](http://arxiv.org/pdf/2502.17832v1)

**Authors**: Hyeonjeong Ha, Qiusi Zhan, Jeonghwan Kim, Dimitrios Bralios, Saikrishna Sanniboina, Nanyun Peng, Kai-wei Chang, Daniel Kang, Heng Ji

**Abstract**: Multimodal large language models (MLLMs) equipped with Retrieval Augmented Generation (RAG) leverage both their rich parametric knowledge and the dynamic, external knowledge to excel in tasks such as Question Answering. While RAG enhances MLLMs by grounding responses in query-relevant external knowledge, this reliance poses a critical yet underexplored safety risk: knowledge poisoning attacks, where misinformation or irrelevant knowledge is intentionally injected into external knowledge bases to manipulate model outputs to be incorrect and even harmful. To expose such vulnerabilities in multimodal RAG, we propose MM-PoisonRAG, a novel knowledge poisoning attack framework with two attack strategies: Localized Poisoning Attack (LPA), which injects query-specific misinformation in both text and images for targeted manipulation, and Globalized Poisoning Attack (GPA) to provide false guidance during MLLM generation to elicit nonsensical responses across all queries. We evaluate our attacks across multiple tasks, models, and access settings, demonstrating that LPA successfully manipulates the MLLM to generate attacker-controlled answers, with a success rate of up to 56% on MultiModalQA. Moreover, GPA completely disrupts model generation to 0% accuracy with just a single irrelevant knowledge injection. Our results highlight the urgent need for robust defenses against knowledge poisoning to safeguard multimodal RAG frameworks.



## **24. Towards Effective Evaluations and Comparisons for LLM Unlearning Methods**

cs.LG

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2406.09179v2) [paper-pdf](http://arxiv.org/pdf/2406.09179v2)

**Authors**: Qizhou Wang, Bo Han, Puning Yang, Jianing Zhu, Tongliang Liu, Masashi Sugiyama

**Abstract**: The imperative to eliminate undesirable data memorization underscores the significance of machine unlearning for large language models (LLMs). Recent research has introduced a series of promising unlearning methods, notably boosting the practical significance of the field. Nevertheless, adopting a proper evaluation framework to reflect the true unlearning efficacy is also essential yet has not received adequate attention. This paper seeks to refine the evaluation of LLM unlearning by addressing two key challenges -- a) the robustness of evaluation metrics and b) the trade-offs between competing goals. The first challenge stems from findings that current metrics are susceptible to various red teaming scenarios. It indicates that they may not reflect the true extent of knowledge retained by LLMs but rather tend to mirror superficial model behaviors, thus prone to attacks. We address this issue by devising and assessing a series of candidate metrics, selecting the most robust ones under various types of attacks. The second challenge arises from the conflicting goals of eliminating unwanted knowledge while retaining those of others. This trade-off between unlearning and retention often fails to conform the Pareto frontier, rendering it subtle to compare the efficacy between methods that excel only in either unlearning or retention. We handle this issue by proposing a calibration method that can restore the original performance on non-targeted data after unlearning, thereby allowing us to focus exclusively on assessing the strength of unlearning. Our evaluation framework notably enhances the effectiveness when assessing and comparing various LLM unlearning methods, further allowing us to benchmark existing works, identify their proper hyper-parameters, and explore new tricks to enhance their practical efficacy.



## **25. Design and implementation of a distributed security threat detection system integrating federated learning and multimodal LLM**

cs.CR

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.17763v1) [paper-pdf](http://arxiv.org/pdf/2502.17763v1)

**Authors**: Yuqing Wang, Xiao Yang

**Abstract**: Traditional security protection methods struggle to address sophisticated attack vectors in large-scale distributed systems, particularly when balancing detection accuracy with data privacy concerns. This paper presents a novel distributed security threat detection system that integrates federated learning with multimodal large language models (LLMs). Our system leverages federated learning to ensure data privacy while employing multimodal LLMs to process heterogeneous data sources including network traffic, system logs, images, and sensor data. Experimental evaluation on a 10TB distributed dataset demonstrates that our approach achieves 96.4% detection accuracy, outperforming traditional baseline models by 4.1 percentage points. The system reduces both false positive and false negative rates by 1.8 and 2.4 percentage points respectively. Performance analysis shows that our system maintains efficient processing capabilities in distributed environments, requiring 180 seconds for model training and 3.8 seconds for threat detection across the distributed network. These results demonstrate significant improvements in detection accuracy and computational efficiency while preserving data privacy, suggesting strong potential for real-world deployment in large-scale security systems.



## **26. Proactive Privacy Amnesia for Large Language Models: Safeguarding PII with Negligible Impact on Model Utility**

cs.CL

ICLR'25 Poster. Project page and code is available at  https://ppa-iclr2025.my.canva.site/

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17591v1) [paper-pdf](http://arxiv.org/pdf/2502.17591v1)

**Authors**: Martin Kuo, Jingyang Zhang, Jianyi Zhang, Minxue Tang, Louis DiValentin, Aolin Ding, Jingwei Sun, William Chen, Amin Hass, Tianlong Chen, Yiran Chen, Hai Li

**Abstract**: With the rise of large language models (LLMs), increasing research has recognized their risk of leaking personally identifiable information (PII) under malicious attacks. Although efforts have been made to protect PII in LLMs, existing methods struggle to balance privacy protection with maintaining model utility. In this paper, inspired by studies of amnesia in cognitive science, we propose a novel approach, Proactive Privacy Amnesia (PPA), to safeguard PII in LLMs while preserving their utility. This mechanism works by actively identifying and forgetting key memories most closely associated with PII in sequences, followed by a memory implanting using suitable substitute memories to maintain the LLM's functionality. We conduct evaluations across multiple models to protect common PII, such as phone numbers and physical addresses, against prevalent PII-targeted attacks, demonstrating the superiority of our method compared with other existing defensive techniques. The results show that our PPA method completely eliminates the risk of phone number exposure by 100% and significantly reduces the risk of physical address exposure by 9.8% - 87.6%, all while maintaining comparable model utility performance.



## **27. The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence**

cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17420v1) [paper-pdf](http://arxiv.org/pdf/2502.17420v1)

**Authors**: Tom Wollschläger, Jannes Elstner, Simon Geisler, Vincent Cohen-Addad, Stephan Günnemann, Johannes Gasteiger

**Abstract**: The safety alignment of large language models (LLMs) can be circumvented through adversarially crafted inputs, yet the mechanisms by which these attacks bypass safety barriers remain poorly understood. Prior work suggests that a single refusal direction in the model's activation space determines whether an LLM refuses a request. In this study, we propose a novel gradient-based approach to representation engineering and use it to identify refusal directions. Contrary to prior work, we uncover multiple independent directions and even multi-dimensional concept cones that mediate refusal. Moreover, we show that orthogonality alone does not imply independence under intervention, motivating the notion of representational independence that accounts for both linear and non-linear effects. Using this framework, we identify mechanistically independent refusal directions. We show that refusal mechanisms in LLMs are governed by complex spatial structures and identify functionally independent directions, confirming that multiple distinct mechanisms drive refusal behavior. Our gradient-based approach uncovers these mechanisms and can further serve as a foundation for future work on understanding LLMs.



## **28. Dataset Featurization: Uncovering Natural Language Features through Unsupervised Data Reconstruction**

cs.AI

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17541v1) [paper-pdf](http://arxiv.org/pdf/2502.17541v1)

**Authors**: Michal Bravansky, Vaclav Kubon, Suhas Hariharan, Robert Kirk

**Abstract**: Interpreting data is central to modern research. Large language models (LLMs) show promise in providing such natural language interpretations of data, yet simple feature extraction methods such as prompting often fail to produce accurate and versatile descriptions for diverse datasets and lack control over granularity and scale. To address these limitations, we propose a domain-agnostic method for dataset featurization that provides precise control over the number of features extracted while maintaining compact and descriptive representations comparable to human expert labeling. Our method optimizes the selection of informative binary features by evaluating the ability of an LLM to reconstruct the original data using those features. We demonstrate its effectiveness in dataset modeling tasks and through two case studies: (1) Constructing a feature representation of jailbreak tactics that compactly captures both the effectiveness and diversity of a larger set of human-crafted attacks; and (2) automating the discovery of features that align with human preferences, achieving accuracy and robustness comparable to expert-crafted features. Moreover, we show that the pipeline scales effectively, improving as additional features are sampled, making it suitable for large and diverse datasets.



## **29. Emoti-Attack: Zero-Perturbation Adversarial Attacks on NLP Systems via Emoji Sequences**

cs.AI

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17392v1) [paper-pdf](http://arxiv.org/pdf/2502.17392v1)

**Authors**: Yangshijie Zhang

**Abstract**: Deep neural networks (DNNs) have achieved remarkable success in the field of natural language processing (NLP), leading to widely recognized applications such as ChatGPT. However, the vulnerability of these models to adversarial attacks remains a significant concern. Unlike continuous domains like images, text exists in a discrete space, making even minor alterations at the sentence, word, or character level easily perceptible to humans. This inherent discreteness also complicates the use of conventional optimization techniques, as text is non-differentiable. Previous research on adversarial attacks in text has focused on character-level, word-level, sentence-level, and multi-level approaches, all of which suffer from inefficiency or perceptibility issues due to the need for multiple queries or significant semantic shifts.   In this work, we introduce a novel adversarial attack method, Emoji-Attack, which leverages the manipulation of emojis to create subtle, yet effective, perturbations. Unlike character- and word-level strategies, Emoji-Attack targets emojis as a distinct layer of attack, resulting in less noticeable changes with minimal disruption to the text. This approach has been largely unexplored in previous research, which typically focuses on emoji insertion as an extension of character-level attacks. Our experiments demonstrate that Emoji-Attack achieves strong attack performance on both large and small models, making it a promising technique for enhancing adversarial robustness in NLP systems.



## **30. REINFORCE Adversarial Attacks on Large Language Models: An Adaptive, Distributional, and Semantic Objective**

cs.LG

30 pages, 6 figures, 15 tables

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17254v1) [paper-pdf](http://arxiv.org/pdf/2502.17254v1)

**Authors**: Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Vincent Cohen-Addad, Johannes Gasteiger, Stephan Günnemann

**Abstract**: To circumvent the alignment of large language models (LLMs), current optimization-based adversarial attacks usually craft adversarial prompts by maximizing the likelihood of a so-called affirmative response. An affirmative response is a manually designed start of a harmful answer to an inappropriate request. While it is often easy to craft prompts that yield a substantial likelihood for the affirmative response, the attacked model frequently does not complete the response in a harmful manner. Moreover, the affirmative objective is usually not adapted to model-specific preferences and essentially ignores the fact that LLMs output a distribution over responses. If low attack success under such an objective is taken as a measure of robustness, the true robustness might be grossly overestimated. To alleviate these flaws, we propose an adaptive and semantic optimization problem over the population of responses. We derive a generally applicable objective via the REINFORCE policy-gradient formalism and demonstrate its efficacy with the state-of-the-art jailbreak algorithms Greedy Coordinate Gradient (GCG) and Projected Gradient Descent (PGD). For example, our objective doubles the attack success rate (ASR) on Llama3 and increases the ASR from 2% to 50% with circuit breaker defense.



## **31. Adversarial Training for Defense Against Label Poisoning Attacks**

cs.LG

Accepted at the International Conference on Learning Representations  (ICLR 2025)

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17121v1) [paper-pdf](http://arxiv.org/pdf/2502.17121v1)

**Authors**: Melis Ilayda Bal, Volkan Cevher, Michael Muehlebach

**Abstract**: As machine learning models grow in complexity and increasingly rely on publicly sourced data, such as the human-annotated labels used in training large language models, they become more vulnerable to label poisoning attacks. These attacks, in which adversaries subtly alter the labels within a training dataset, can severely degrade model performance, posing significant risks in critical applications. In this paper, we propose FLORAL, a novel adversarial training defense strategy based on support vector machines (SVMs) to counter these threats. Utilizing a bilevel optimization framework, we cast the training process as a non-zero-sum Stackelberg game between an attacker, who strategically poisons critical training labels, and the model, which seeks to recover from such attacks. Our approach accommodates various model architectures and employs a projected gradient descent algorithm with kernel SVMs for adversarial training. We provide a theoretical analysis of our algorithm's convergence properties and empirically evaluate FLORAL's effectiveness across diverse classification tasks. Compared to robust baselines and foundation models such as RoBERTa, FLORAL consistently achieves higher robust accuracy under increasing attacker budgets. These results underscore the potential of FLORAL to enhance the resilience of machine learning models against label poisoning threats, thereby ensuring robust classification in adversarial settings.



## **32. GuidedBench: Equipping Jailbreak Evaluation with Guidelines**

cs.CL

Homepage: https://sproutnan.github.io/AI-Safety_Benchmark/

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.16903v1) [paper-pdf](http://arxiv.org/pdf/2502.16903v1)

**Authors**: Ruixuan Huang, Xunguang Wang, Zongjie Li, Daoyuan Wu, Shuai Wang

**Abstract**: Jailbreaking methods for large language models (LLMs) have gained increasing attention for building safe and responsible AI systems. After analyzing 35 jailbreak methods across six categories, we find that existing benchmarks, relying on universal LLM-based or keyword-matching scores, lack case-specific criteria, leading to conflicting results. In this paper, we introduce a more robust evaluation framework for jailbreak methods, with a curated harmful question dataset, detailed case-by-case evaluation guidelines, and a scoring system equipped with these guidelines. Our experiments show that existing jailbreak methods exhibit better discrimination when evaluated using our benchmark. Some jailbreak methods that claim to achieve over 90% attack success rate (ASR) on other benchmarks only reach a maximum of 30.2% on our benchmark, providing a higher ceiling for more advanced jailbreak research; furthermore, using our scoring system reduces the variance of disagreements between different evaluator LLMs by up to 76.33%. This demonstrates its ability to provide more fair and stable evaluation.



## **33. Char-mander Use mBackdoor! A Study of Cross-lingual Backdoor Attacks in Multilingual LLMs**

cs.CL

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.16901v1) [paper-pdf](http://arxiv.org/pdf/2502.16901v1)

**Authors**: Himanshu Beniwal, Sailesh Panda, Mayank Singh

**Abstract**: We explore Cross-lingual Backdoor ATtacks (X-BAT) in multilingual Large Language Models (mLLMs), revealing how backdoors inserted in one language can automatically transfer to others through shared embedding spaces. Using toxicity classification as a case study, we demonstrate that attackers can compromise multilingual systems by poisoning data in a single language, with rare tokens serving as specific effective triggers. Our findings expose a critical vulnerability in the fundamental architecture that enables cross-lingual transfer in these models. Our code and data are publicly available at https://github.com/himanshubeniwal/X-BAT.



## **34. PAPILLON: Efficient and Stealthy Fuzz Testing-Powered Jailbreaks for LLMs**

cs.CR

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2409.14866v4) [paper-pdf](http://arxiv.org/pdf/2409.14866v4)

**Authors**: Xueluan Gong, Mingzhe Li, Yilin Zhang, Fengyuan Ran, Chen Chen, Yanjiao Chen, Qian Wang, Kwok-Yan Lam

**Abstract**: Large Language Models (LLMs) have excelled in various tasks but are still vulnerable to jailbreaking attacks, where attackers create jailbreak prompts to mislead the model to produce harmful or offensive content. Current jailbreak methods either rely heavily on manually crafted templates, which pose challenges in scalability and adaptability, or struggle to generate semantically coherent prompts, making them easy to detect. Additionally, most existing approaches involve lengthy prompts, leading to higher query costs.In this paper, to remedy these challenges, we introduce a novel jailbreaking attack framework called PAPILLON, which is an automated, black-box jailbreaking attack framework that adapts the black-box fuzz testing approach with a series of customized designs. Instead of relying on manually crafted templates,PAPILLON starts with an empty seed pool, removing the need to search for any related jailbreaking templates. We also develop three novel question-dependent mutation strategies using an LLM helper to generate prompts that maintain semantic coherence while significantly reducing their length. Additionally, we implement a two-level judge module to accurately detect genuine successful jailbreaks. We evaluated PAPILLON on 7 representative LLMs and compared it with 5 state-of-the-art jailbreaking attack strategies. For proprietary LLM APIs, such as GPT-3.5 turbo, GPT-4, and Gemini-Pro, PAPILLONs achieves attack success rates of over 90%, 80%, and 74%, respectively, exceeding existing baselines by more than 60\%. Additionally, PAPILLON can maintain high semantic coherence while significantly reducing the length of jailbreak prompts. When targeting GPT-4, PAPILLON can achieve over 78% attack success rate even with 100 tokens. Moreover, PAPILLON demonstrates transferability and is robust to state-of-the-art defenses.



## **35. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

cs.CV

Accepted by ICLR2025

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2406.18849v4) [paper-pdf](http://arxiv.org/pdf/2406.18849v4)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 24 advanced open-source LVLMs and 2 close-source LVLMs are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released at https://github.com/Robin-WZQ/Dysca.



## **36. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16750v1) [paper-pdf](http://arxiv.org/pdf/2502.16750v1)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehnaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.



## **37. RapidPen: Fully Automated IP-to-Shell Penetration Testing with LLM-based Agents**

cs.CR

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16730v1) [paper-pdf](http://arxiv.org/pdf/2502.16730v1)

**Authors**: Sho Nakatani

**Abstract**: We present RapidPen, a fully automated penetration testing (pentesting) framework that addresses   the challenge of achieving an initial foothold (IP-to-Shell) without human intervention. Unlike prior   approaches that focus primarily on post-exploitation or require a human-in-the-loop, RapidPen   leverages large language models (LLMs) to autonomously discover and exploit vulnerabilities, starting from   a single IP address. By integrating advanced ReAct-style task planning (Re) with retrieval-augmented   knowledge bases of successful exploits, along with a command-generation and direct execution feedback loop   (Act), RapidPen systematically scans services, identifies viable attack vectors, and executes targeted   exploits in a fully automated manner.   In our evaluation against a vulnerable target from the Hack The Box platform, RapidPen achieved shell   access within 200-400 seconds at a per-run cost of approximately \$0.3-\$0.6, demonstrating a   60\% success rate when reusing prior "success-case" data. These results underscore the potential   of truly autonomous pentesting for both security novices and seasoned professionals. Organizations   without dedicated security teams can leverage RapidPen to quickly identify critical vulnerabilities,   while expert pentesters can offload repetitive tasks and focus on complex challenges.   Ultimately, our work aims to make penetration testing more accessible and cost-efficient,   thereby enhancing the overall security posture of modern software ecosystems.



## **38. Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images**

cs.AI

Accepted to ICLR 2025

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16593v1) [paper-pdf](http://arxiv.org/pdf/2502.16593v1)

**Authors**: Yubo Wang, Jianting Tang, Chaohu Liu, Linli Xu

**Abstract**: Large vision-language models (LVLMs) have demonstrated remarkable image understanding and dialogue capabilities, allowing them to handle a variety of visual question answering tasks. However, their widespread availability raises concerns about unauthorized usage and copyright infringement, where users or individuals can develop their own LVLMs by fine-tuning published models. In this paper, we propose a novel method called Parameter Learning Attack (PLA) for tracking the copyright of LVLMs without modifying the original model. Specifically, we construct adversarial images through targeted attacks against the original model, enabling it to generate specific outputs. To ensure these attacks remain effective on potential fine-tuned models to trigger copyright tracking, we allow the original model to learn the trigger images by updating parameters in the opposite direction during the adversarial attack process. Notably, the proposed method can be applied after the release of the original model, thus not affecting the model's performance and behavior. To simulate real-world applications, we fine-tune the original model using various strategies across diverse datasets, creating a range of models for copyright verification. Extensive experiments demonstrate that our method can more effectively identify the original copyright of fine-tuned models compared to baseline methods. Therefore, this work provides a powerful tool for tracking copyrights and detecting unlicensed usage of LVLMs.



## **39. Can Indirect Prompt Injection Attacks Be Detected and Removed?**

cs.CR

17 pages, 6 figures

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16580v1) [paper-pdf](http://arxiv.org/pdf/2502.16580v1)

**Authors**: Yulin Chen, Haoran Li, Yuan Sui, Yufei He, Yue Liu, Yangqiu Song, Bryan Hooi

**Abstract**: Prompt injection attacks manipulate large language models (LLMs) by misleading them to deviate from the original input instructions and execute maliciously injected instructions, because of their instruction-following capabilities and inability to distinguish between the original input instructions and maliciously injected instructions. To defend against such attacks, recent studies have developed various detection mechanisms. While significant efforts have focused on detecting direct prompt injection attacks, where injected instructions are directly from the attacker who is also the user, limited attention has been given to indirect prompt injection attacks, where injected instructions are indirectly from external tools, such as a search engine. Moreover, current works mainly investigate injection detection methods and pay less attention to the post-processing method that aims to mitigate the injection after detection. In this paper, we investigate the feasibility of detecting and removing indirect prompt injection attacks, and we construct a benchmark dataset for evaluation. For detection, we assess the performance of existing LLMs and open-source detection models, and we further train detection models using our crafted training datasets. For removal, we evaluate two intuitive methods: (1) the segmentation removal method, which segments the injected document and removes parts containing injected instructions, and (2) the extraction removal method, which trains an extraction model to identify and remove injected instructions.



## **40. SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model**

cs.CR

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2501.18636v2) [paper-pdf](http://arxiv.org/pdf/2501.18636v2)

**Authors**: Xun Liang, Simin Niu, Zhiyu Li, Sensen Zhang, Hanyu Wang, Feiyu Xiong, Jason Zhaoxin Fan, Bo Tang, Shichao Song, Mengwei Wang, Jiawei Yang

**Abstract**: The indexing-retrieval-generation paradigm of retrieval-augmented generation (RAG) has been highly successful in solving knowledge-intensive tasks by integrating external knowledge into large language models (LLMs). However, the incorporation of external and unverified knowledge increases the vulnerability of LLMs because attackers can perform attack tasks by manipulating knowledge. In this paper, we introduce a benchmark named SafeRAG designed to evaluate the RAG security. First, we classify attack tasks into silver noise, inter-context conflict, soft ad, and white Denial-of-Service. Next, we construct RAG security evaluation dataset (i.e., SafeRAG dataset) primarily manually for each task. We then utilize the SafeRAG dataset to simulate various attack scenarios that RAG may encounter. Experiments conducted on 14 representative RAG components demonstrate that RAG exhibits significant vulnerability to all attack tasks and even the most apparent attack task can easily bypass existing retrievers, filters, or advanced LLMs, resulting in the degradation of RAG service quality. Code is available at: https://github.com/IAAR-Shanghai/SafeRAG.



## **41. On Calibration of LLM-based Guard Models for Reliable Content Moderation**

cs.CR

Accepted to ICLR 2025

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2410.10414v2) [paper-pdf](http://arxiv.org/pdf/2410.10414v2)

**Authors**: Hongfu Liu, Hengguan Huang, Xiangming Gu, Hao Wang, Ye Wang

**Abstract**: Large language models (LLMs) pose significant risks due to the potential for generating harmful content or users attempting to evade guardrails. Existing studies have developed LLM-based guard models designed to moderate the input and output of threat LLMs, ensuring adherence to safety policies by blocking content that violates these protocols upon deployment. However, limited attention has been given to the reliability and calibration of such guard models. In this work, we empirically conduct comprehensive investigations of confidence calibration for 9 existing LLM-based guard models on 12 benchmarks in both user input and model output classification. Our findings reveal that current LLM-based guard models tend to 1) produce overconfident predictions, 2) exhibit significant miscalibration when subjected to jailbreak attacks, and 3) demonstrate limited robustness to the outputs generated by different types of response models. Additionally, we assess the effectiveness of post-hoc calibration methods to mitigate miscalibration. We demonstrate the efficacy of temperature scaling and, for the first time, highlight the benefits of contextual calibration for confidence calibration of guard models, particularly in the absence of validation sets. Our analysis and experiments underscore the limitations of current LLM-based guard models and provide valuable insights for the future development of well-calibrated guard models toward more reliable content moderation. We also advocate for incorporating reliability evaluation of confidence calibration when releasing future LLM-based guard models.



## **42. Intrinsic Model Weaknesses: How Priming Attacks Unveil Vulnerabilities in Large Language Models**

cs.CL

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16491v1) [paper-pdf](http://arxiv.org/pdf/2502.16491v1)

**Authors**: Yuyi Huang, Runzhe Zhan, Derek F. Wong, Lidia S. Chao, Ailin Tao

**Abstract**: Large language models (LLMs) have significantly influenced various industries but suffer from a critical flaw, the potential sensitivity of generating harmful content, which poses severe societal risks. We developed and tested novel attack strategies on popular LLMs to expose their vulnerabilities in generating inappropriate content. These strategies, inspired by psychological phenomena such as the "Priming Effect", "Safe Attention Shift", and "Cognitive Dissonance", effectively attack the models' guarding mechanisms. Our experiments achieved an attack success rate (ASR) of 100% on various open-source models, including Meta's Llama-3.2, Google's Gemma-2, Mistral's Mistral-NeMo, Falcon's Falcon-mamba, Apple's DCLM, Microsoft's Phi3, and Qwen's Qwen2.5, among others. Similarly, for closed-source models such as OpenAI's GPT-4o, Google's Gemini-1.5, and Claude-3.5, we observed an ASR of at least 95% on the AdvBench dataset, which represents the current state-of-the-art. This study underscores the urgent need to reassess the use of generative models in critical applications to mitigate potential adverse societal impacts.



## **43. Swallowing the Poison Pills: Insights from Vulnerability Disparity Among LLMs**

cs.CR

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.18518v1) [paper-pdf](http://arxiv.org/pdf/2502.18518v1)

**Authors**: Peng Yifeng, Wu Zhizheng, Chen Chen

**Abstract**: Modern large language models (LLMs) exhibit critical vulnerabilities to poison pill attacks: localized data poisoning that alters specific factual knowledge while preserving overall model utility. We systematically demonstrate these attacks exploit inherent architectural properties of LLMs, achieving 54.6% increased retrieval inaccuracy on long-tail knowledge versus dominant topics and up to 25.5% increase retrieval inaccuracy on compressed models versus original architectures. Through controlled mutations (e.g., temporal/spatial/entity alterations) and, our method induces localized memorization deterioration with negligible impact on models' performance on regular standard benchmarks (e.g., <2% performance drop on MMLU/GPQA), leading to potential detection evasion. Our findings suggest: (1) Disproportionate vulnerability in long-tail knowledge may result from reduced parameter redundancy; (2) Model compression may increase attack surfaces, with pruned/distilled models requiring 30% fewer poison samples for equivalent damage; (3) Associative memory enables both spread of collateral damage to related concepts and amplification of damage from simultaneous attack, particularly for dominant topics. These findings raise concerns over current scaling paradigms since attack costs are lowering while defense complexity is rising. Our work establishes poison pills as both a security threat and diagnostic tool, revealing critical security-efficiency trade-offs in language model compression that challenges prevailing safety assumptions.



## **44. A generative approach to LLM harmfulness detection with special red flag tokens**

cs.CL

13 pages, 6 figures

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16366v1) [paper-pdf](http://arxiv.org/pdf/2502.16366v1)

**Authors**: Sophie Xhonneux, David Dobre, Mehrnaz Mohfakhami, Leo Schwinn, Gauthier Gidel

**Abstract**: Most safety training methods for large language models (LLMs) based on fine-tuning rely on dramatically changing the output distribution of the model when faced with a harmful request, shifting it from an unsafe answer to a refusal to respond. These methods inherently compromise model capabilities and might make auto-regressive models vulnerable to attacks that make likely an initial token of affirmative response. To avoid that, we propose to expand the model's vocabulary with a special token we call red flag token (<rf>) and propose to fine-tune the model to generate this token at any time harmful content is generated or about to be generated. This novel safety training method effectively augments LLMs into generative classifiers of harmfulness at all times during the conversation. This method offers several advantages: it enables the model to explicitly learn the concept of harmfulness while marginally affecting the generated distribution, thus maintaining the model's utility. It also evaluates each generated answer rather than just the input prompt and provides a stronger defence against sampling-based attacks. In addition, it simplifies the evaluation of the model's robustness and reduces correlated failures when combined with a classifier. We further show an increased robustness to long contexts, and supervised fine-tuning attacks.



## **45. ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models**

cs.CR

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.18511v1) [paper-pdf](http://arxiv.org/pdf/2502.18511v1)

**Authors**: Xuxu Liu, Siyuan Liang, Mengya Han, Yong Luo, Aishan Liu, Xiantao Cai, Zheng He, Dacheng Tao

**Abstract**: Generative large language models are crucial in natural language processing, but they are vulnerable to backdoor attacks, where subtle triggers compromise their behavior. Although backdoor attacks against LLMs are constantly emerging, existing benchmarks remain limited in terms of sufficient coverage of attack, metric system integrity, backdoor attack alignment. And existing pre-trained backdoor attacks are idealized in practice due to resource access constraints. Therefore we establish $\textit{ELBA-Bench}$, a comprehensive and unified framework that allows attackers to inject backdoor through parameter efficient fine-tuning ($\textit{e.g.,}$ LoRA) or without fine-tuning techniques ($\textit{e.g.,}$ In-context-learning). $\textit{ELBA-Bench}$ provides over 1300 experiments encompassing the implementations of 12 attack methods, 18 datasets, and 12 LLMs. Extensive experiments provide new invaluable findings into the strengths and limitations of various attack strategies. For instance, PEFT attack consistently outperform without fine-tuning approaches in classification tasks while showing strong cross-dataset generalization with optimized triggers boosting robustness; Task-relevant backdoor optimization techniques or attack prompts along with clean and adversarial demonstrations can enhance backdoor attack success while preserving model performance on clean samples. Additionally, we introduce a universal toolbox designed for standardized backdoor attack research, with the goal of propelling further progress in this vital area.



## **46. Na'vi or Knave: Jailbreaking Language Models via Metaphorical Avatars**

cs.CL

Our study requires further in-depth research to ensure the  comprehensiveness and adequacy of the methodology

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2412.12145v4) [paper-pdf](http://arxiv.org/pdf/2412.12145v4)

**Authors**: Yu Yan, Sheng Sun, Junqi Tong, Min Liu, Qi Li

**Abstract**: Metaphor serves as an implicit approach to convey information, while enabling the generalized comprehension of complex subjects. However, metaphor can potentially be exploited to bypass the safety alignment mechanisms of Large Language Models (LLMs), leading to the theft of harmful knowledge. In our study, we introduce a novel attack framework that exploits the imaginative capacity of LLMs to achieve jailbreaking, the J\underline{\textbf{A}}ilbreak \underline{\textbf{V}}ia \underline{\textbf{A}}dversarial Me\underline{\textbf{TA}} -pho\underline{\textbf{R}} (\textit{AVATAR}). Specifically, to elicit the harmful response, AVATAR extracts harmful entities from a given harmful target and maps them to innocuous adversarial entities based on LLM's imagination. Then, according to these metaphors, the harmful target is nested within human-like interaction for jailbreaking adaptively. Experimental results demonstrate that AVATAR can effectively and transferablly jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. Our study exposes a security risk in LLMs from their endogenous imaginative capabilities. Furthermore, the analytical study reveals the vulnerability of LLM to adversarial metaphors and the necessity of developing defense methods against jailbreaking caused by the adversarial metaphor. \textcolor{orange}{ \textbf{Warning: This paper contains potentially harmful content from LLMs.}}



## **47. Humanizing the Machine: Proxy Attacks to Mislead LLM Detectors**

cs.LG

29 pages

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2410.19230v2) [paper-pdf](http://arxiv.org/pdf/2410.19230v2)

**Authors**: Tianchun Wang, Yuanzhou Chen, Zichuan Liu, Zhanwen Chen, Haifeng Chen, Xiang Zhang, Wei Cheng

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of text generation, producing outputs that closely mimic human-like writing. Although academic and industrial institutions have developed detectors to prevent the malicious usage of LLM-generated texts, other research has doubt about the robustness of these systems. To stress test these detectors, we introduce a proxy-attack strategy that effortlessly compromises LLMs, causing them to produce outputs that align with human-written text and mislead detection systems. Our method attacks the source model by leveraging a reinforcement learning (RL) fine-tuned humanized small language model (SLM) in the decoding phase. Through an in-depth analysis, we demonstrate that our attack strategy is capable of generating responses that are indistinguishable to detectors, preventing them from differentiating between machine-generated and human-written text. We conduct systematic evaluations on extensive datasets using proxy-attacked open-source models, including Llama2-13B, Llama3-70B, and Mixtral-8*7B in both white- and black-box settings. Our findings show that the proxy-attack strategy effectively deceives the leading detectors, resulting in an average AUROC drop of 70.4% across multiple datasets, with a maximum drop of 90.3% on a single dataset. Furthermore, in cross-discipline scenarios, our strategy also bypasses these detectors, leading to a significant relative decrease of up to 90.9%, while in cross-language scenario, the drop reaches 91.3%. Despite our proxy-attack strategy successfully bypassing the detectors with such significant relative drops, we find that the generation quality of the attacked models remains preserved, even within a modest utility budget, when compared to the text produced by the original, unattacked source model.



## **48. Be a Multitude to Itself: A Prompt Evolution Framework for Red Teaming**

cs.CL

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16109v1) [paper-pdf](http://arxiv.org/pdf/2502.16109v1)

**Authors**: Rui Li, Peiyi Wang, Jingyuan Ma, Di Zhang, Lei Sha, Zhifang Sui

**Abstract**: Large Language Models (LLMs) have gained increasing attention for their remarkable capacity, alongside concerns about safety arising from their potential to produce harmful content. Red teaming aims to find prompts that could elicit harmful responses from LLMs, and is essential to discover and mitigate safety risks before real-world deployment. However, manual red teaming is both time-consuming and expensive, rendering it unscalable. In this paper, we propose RTPE, a scalable evolution framework to evolve red teaming prompts across both breadth and depth dimensions, facilitating the automatic generation of numerous high-quality and diverse red teaming prompts. Specifically, in-breadth evolving employs a novel enhanced in-context learning method to create a multitude of quality prompts, whereas in-depth evolving applies customized transformation operations to enhance both content and form of prompts, thereby increasing diversity. Extensive experiments demonstrate that RTPE surpasses existing representative automatic red teaming methods on both attack success rate and diversity. In addition, based on 4,800 red teaming prompts created by RTPE, we further provide a systematic analysis of 8 representative LLMs across 8 sensitive topics.



## **49. Merger-as-a-Stealer: Stealing Targeted PII from Aligned LLMs with Model Merging**

cs.CR

17 pages, 3 figures

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16094v1) [paper-pdf](http://arxiv.org/pdf/2502.16094v1)

**Authors**: Lin Lu, Zhigang Zuo, Ziji Sheng, Pan Zhou

**Abstract**: Model merging has emerged as a promising approach for updating large language models (LLMs) by integrating multiple domain-specific models into a cross-domain merged model. Despite its utility and plug-and-play nature, unmonitored mergers can introduce significant security vulnerabilities, such as backdoor attacks and model merging abuse. In this paper, we identify a novel and more realistic attack surface where a malicious merger can extract targeted personally identifiable information (PII) from an aligned model with model merging. Specifically, we propose \texttt{Merger-as-a-Stealer}, a two-stage framework to achieve this attack: First, the attacker fine-tunes a malicious model to force it to respond to any PII-related queries. The attacker then uploads this malicious model to the model merging conductor and obtains the merged model. Second, the attacker inputs direct PII-related queries to the merged model to extract targeted PII. Extensive experiments demonstrate that \texttt{Merger-as-a-Stealer} successfully executes attacks against various LLMs and model merging methods across diverse settings, highlighting the effectiveness of the proposed framework. Given that this attack enables character-level extraction for targeted PII without requiring any additional knowledge from the attacker, we stress the necessity for improved model alignment and more robust defense mechanisms to mitigate such threats.



## **50. Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack**

cs.CR

12 pages, 5 figures

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16086v1) [paper-pdf](http://arxiv.org/pdf/2502.16086v1)

**Authors**: Chenxi Dai, Lin Lu, Pan Zhou

**Abstract**: Decentralized training has become a resource-efficient framework to democratize the training of large language models (LLMs). However, the privacy risks associated with this framework, particularly due to the potential inclusion of sensitive data in training datasets, remain unexplored. This paper identifies a novel and realistic attack surface: the privacy leakage from training data in decentralized training, and proposes \textit{activation inversion attack} (AIA) for the first time. AIA first constructs a shadow dataset comprising text labels and corresponding activations using public datasets. Leveraging this dataset, an attack model can be trained to reconstruct the training data from activations in victim decentralized training. We conduct extensive experiments on various LLMs and publicly available datasets to demonstrate the susceptibility of decentralized training to AIA. These findings highlight the urgent need to enhance security measures in decentralized training to mitigate privacy risks in training LLMs.



