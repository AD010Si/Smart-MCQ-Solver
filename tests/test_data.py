from mcq_solver.data import add_option_set, grouped_split


def test_no_option_set_overlap(toy_df):
    kb, ev = grouped_split(toy_df, test_size=0.2, seed=42)
    assert len(kb) + len(ev) == len(toy_df)
    assert not set(kb["options_set"]) & set(ev["options_set"])


def test_split_is_deterministic(toy_df):
    a = grouped_split(toy_df)[1]["id"].tolist()
    b = grouped_split(toy_df)[1]["id"].tolist()
    assert a == b


def test_options_set_column(toy_df):
    assert "options_set" in add_option_set(toy_df)
