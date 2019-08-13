class Hparams:
    batch_size = 128
    enc_maxlen = 20
    dec_maxlen = 20
    num_epochs = 20
    hidden_units = 128
    emb_units = 64
    graphemes = ["<pad>", "<unk>", "</s>"] + list(" ًٌٍَُِآئابتثجحخدذرزسشصضطظعغفقلمنهوپچژکگی")
    phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + list("?ACSZ^_abdefghijklmnopqrstuvxyz")
    lr = 0.001

hp = Hparams()