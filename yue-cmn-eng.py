import datasets


_LANGUAGE_PAIRS = [("eng", "yue"), ("eng", "cmn"), ("cmn", "yue")]

class YueCmnEngConfig(datasets.BuilderConfig):
    """BuilderConfig for YueCmnEng dataset."""
    
    def __init__(self, *args, language_pair=(None, None), **kwargs):
        super().__init__(*args, **kwargs)
        self.language_pair = language_pair

class YueCmnEng(datasets.GeneratorBasedBuilder):
    """YueCmnEng dataset."""
    
    BUILDER_CONFIGS = [
        YueCmnEngConfig(
            name=f"{l1}_{l2}",
            description=f"Translation dataset between {l1} and {l2}.",
            language_pair=(l1, l2),
            version=datasets.Version("1.0.0"),
        )
        for l1, l2 in _LANGUAGE_PAIRS
    ]
    
    def _info(self):
        return datasets.DatasetInfo(
            description="Translations",
            features=datasets.Features(
                {
                    "translation": datasets.Translation(languages=self.config.language_pair),
                }
            ),
            supervised_keys=self.config.language_pair,
            citation="TODO",
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"language_pair": self.config.language_pair}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"language_pair": self.config.language_pair}),
        ]

    def _generate_examples(self, language_pair):
        l1, l2 = language_pair
        for i, (l1_sent, l2_sent) in enumerate(zip(l1, l2)):
            yield i, {
                "translation": {l1: l1_sent, l2: l2_sent},
            }
