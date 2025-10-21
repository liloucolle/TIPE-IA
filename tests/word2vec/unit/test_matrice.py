
from word2vec.matrice import softmax, descente_de_gradient, epoche, entrainer, perte_tot, deep_learning
from text_utils.text_utils import tokenisation, vocabulaire, creation_paires, frequence_mots 
import numpy as np
import pytest

# On suppose la fonction:
# def softmax(matrice_centre, matrice_contexte, id_mot_centre):
#     v_c = matrice_centre[id_mot_centre]
#     s_j = np.dot(matrice_contexte, v_c)
#     exp_sj = np.exp(s_j - np.max(s_j))
#     return exp_sj / np.sum(exp_sj)

class TestSoftmax:
    """Unit tests for the efficient softmax implementation"""

    def test_basic_three_values(self):
        """Distribution correcte (trois valeurs) et somme = 1"""
        matrice_centre = np.array([
            [0.2, 0.8, 0.5],   # id 0
            [0.1, 0.3, 0.9],   # id 1
            [0.7, 0.2, 0.4]    # id 2
        ])
        matrice_contexte = np.array([
            [0.5, 0.1, 0.3],
            [0.9, 0.7, 0.2],
            [0.2, 0.4, 0.8]
        ])

        res = softmax(matrice_centre, matrice_contexte, 0)
        assert res.shape == (3,)
        assert np.all(res >= 0)
        assert np.isclose(np.sum(res), 1.0, atol=1e-8)

        # Vérif avec calcul "attendu"
        logits = matrice_contexte @ matrice_centre[0]
        expected = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
        assert np.allclose(res, expected, atol=1e-8)

    def test_uniform_when_equal_logits(self):
        """Si tous les produits scalaires sont égaux → proba uniforme"""
        # même contexte, même v_c ⇒ mêmes logits
        matrice_centre = np.array([[1.0, 1.0], [2.0, 2.0]])
        matrice_contexte = np.array([[3.0, 4.0],
                                     [3.0, 4.0],
                                     [3.0, 4.0]])
        res = softmax(matrice_centre, matrice_contexte, 0)
        assert np.allclose(res, np.array([1/3, 1/3, 1/3]), atol=1e-8)

    def test_zero_center_vector(self):
        """v_c = 0 ⇒ logits tous nuls ⇒ proba uniforme"""
        matrice_centre = np.array([[0.0, 0.0]])
        matrice_contexte = np.random.rand(5, 2)
        res = softmax(matrice_centre, matrice_contexte, 0)
        assert np.allclose(res, np.full(5, 1/5), atol=1e-8)

    def test_numerical_stability_large_values(self):
        """Robustesse pour grandes valeurs (pas d'overflow)"""
        matrice_centre = np.full((2, 3), 1e3)
        matrice_contexte = np.full((4, 3), 1e3)
        res = softmax(matrice_centre, matrice_contexte, 1)
        # Les 4 logits sont égaux ⇒ uniforme
        assert np.allclose(res, np.full(4, 1/4), atol=1e-8)

    def test_shape_matches_num_context_rows(self):
        """La longueur de sortie = nb de lignes du contexte"""
        C, D = 7, 10
        matrice_centre = np.random.randn(C, D)
        matrice_contexte = np.random.randn(13, D)
        res = softmax(matrice_centre, matrice_contexte, 3)
        assert res.shape == (13,)

    def test_invalid_center_index_raises(self):
        """Index centre hors bornes ⇒ IndexError"""
        matrice_centre = np.random.randn(3, 5)
        matrice_contexte = np.random.randn(4, 5)
        with pytest.raises(IndexError):
            _ = softmax(matrice_centre, matrice_contexte, 10)

    def test_mismatched_dimensions_raises(self):
        """Dimensions incompatibles pour le produit scalaire ⇒ ValueError (NumPy)"""
        matrice_centre = np.random.randn(4, 8)   # dim 8
        matrice_contexte = np.random.randn(5, 6) # dim 6 (incompatible)
        with pytest.raises(ValueError):
            _ = softmax(matrice_centre, matrice_contexte, 2)

    def test_probabilities_change_with_center(self):
        """Changer le mot centre change la distribution (sauf cas dégénérés)"""
        matrice_centre = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        matrice_contexte = np.array([
            [1.0, 0.0, 0.0],   # favorise centre 0
            [0.0, 1.0, 0.0],   # favorise centre 1
            [0.5, 0.5, 0.0]    # neutre entre 0 et 1
        ])
        p0 = softmax(matrice_centre, matrice_contexte, 0)
        p1 = softmax(matrice_centre, matrice_contexte, 1)
        assert not np.allclose(p0, p1)
        assert np.isclose(np.sum(p0), 1.0)
        assert np.isclose(np.sum(p1), 1.0)

    def test_returns_numpy_array(self):
        """Le type de retour est un ndarray (pas une liste)"""
        matrice_centre = np.random.randn(2, 3)
        matrice_contexte = np.random.randn(3, 3)
        res = softmax(matrice_centre, matrice_contexte, 1)
        assert isinstance(res, np.ndarray)




class TestDescenteGradient:
    """Tests unitaires pour la fonction descente_de_gradient"""

    def test_update_reduit_valeur_grad(self):
        """Test que la mise à jour va dans la direction du gradient (descend la perte)"""
        np.random.seed(0)
        matrice_centre = np.random.randn(3, 4)
        matrice_contexte = np.random.randn(3, 4)
        proba = np.array([0.2, 0.5, 0.3])
        id_centre = 1
        id_pos = 2
        eta = 0.1

        # gradient attendu : M^T (p - y)
        y = np.zeros(3); y[id_pos] = 1.0
        grad_attendu = np.matmul(matrice_contexte.T, (proba - y))
        vc_avant = matrice_centre[id_centre].copy()

        matrice_centre = descente_de_gradient(matrice_centre, matrice_contexte,
                                              proba, id_centre, id_pos, eta)

        vc_apres = matrice_centre[id_centre]
        direction = np.dot(grad_attendu, vc_apres - vc_avant)
        # le produit scalaire doit être négatif (descente)
        assert direction < 0

    def test_aucune_modif_si_pas_d_erreur(self):
        """Si proba == y, le gradient vaut 0 donc pas de changement"""
        matrice_centre = np.random.randn(3, 4)
        matrice_contexte = np.random.randn(3, 4)
        proba = np.array([0.0, 1.0, 0.0])  # bonne prédiction parfaite
        id_centre = 0
        id_pos = 1
        eta = 0.1

        avant_centre = matrice_centre.copy()
        descente_de_gradient(matrice_centre, matrice_contexte, proba,
                             id_centre, id_pos, eta)

        assert np.allclose(matrice_centre, avant_centre, atol=1e-10)

    def test_mise_a_jour_depend_du_taux(self):
        """Vérifie que le pas de descente dépend bien du taux d'apprentissage"""
        matrice_centre = np.random.randn(2, 3)
        matrice_contexte = np.random.randn(2, 3)
        proba = np.array([0.4, 0.6])
        id_centre = 0
        id_pos = 1
        eta1, eta2 = 0.1, 0.5

        mc1 = matrice_centre.copy()
        mc2 = matrice_centre.copy()

        descente_de_gradient(mc1, matrice_contexte, proba, id_centre, id_pos, eta1)
        descente_de_gradient(mc2, matrice_contexte, proba, id_centre, id_pos, eta2)

        # la mise à jour avec eta2 doit être plus grande
        delta1 = np.linalg.norm(mc1[id_centre] - matrice_centre[id_centre])
        delta2 = np.linalg.norm(mc2[id_centre] - matrice_centre[id_centre])
        assert delta2 > delta1

    def test_output_type(self):
        """Le type de retour doit être un np.ndarray"""
        matrice_centre = np.random.randn(3, 3)
        matrice_contexte = np.random.randn(3, 3)
        proba = np.array([0.2, 0.5, 0.3])
        id_centre = 1
        id_pos = 0
        eta = 0.05
        res = descente_de_gradient(matrice_centre, matrice_contexte, proba,
                                   id_centre, id_pos, eta)
        assert isinstance(res, np.ndarray)

    def test_forme_matrice_inchangee(self):
        """La forme des matrices doit rester identique"""
        matrice_centre = np.random.randn(3, 5)
        matrice_contexte = np.random.randn(3, 5)
        proba = np.array([0.3, 0.4, 0.3])
        id_centre = 0
        id_pos = 2
        eta = 0.1

        shape_centre = matrice_centre.shape
        shape_contexte = matrice_contexte.shape

        descente_de_gradient(matrice_centre, matrice_contexte, proba,
                             id_centre, id_pos, eta)

        assert matrice_centre.shape == shape_centre
        assert matrice_contexte.shape == shape_contexte



# --- Helpers pour les tests ---
def softmax_scores(M, v_c):
    s = M @ v_c
    e = np.exp(s - np.max(s))
    return e / np.sum(e)

def loss_total(matrice_centre, matrice_contexte, paires):
    """Somme des pertes -log P(context|centre) sur toutes les paires."""
    total = 0.0
    for (id_c, id_pos) in paires:
        v_c = matrice_centre[id_c]
        p = softmax_scores(matrice_contexte, v_c)
        total += -np.log(p[id_pos])
    return float(total)


class TestEpoche:
    def test_returns_tuple_and_shapes(self):
        """epoche doit retourner (matrice_centre, matrice_contexte) de même shape qu'en entrée."""
        C, V, d = 4, 6, 3
        mc = np.random.randn(C, d)
        mx = np.random.randn(V, d)
        paires = [(0, 1), (1, 2), (2, 3), (3, 4)]

        mc_shape, mx_shape = mc.shape, mx.shape
        mc_out, mx_out = epoche(mc, mx, paires, taux_apprentissage=0.05)

        assert isinstance(mc_out, np.ndarray) and isinstance(mx_out, np.ndarray)
        assert mc_out.shape == mc_shape
        assert mx_out.shape == mx_shape

    def test_zero_learning_rate_no_change(self):
        """Avec eta=0, aucune mise à jour ne doit avoir lieu (matrices inchangées)."""
        C, V, d = 3, 5, 4
        mc = np.random.randn(C, d)
        mx = np.random.randn(V, d)
        paires = [(0, 1), (1, 2), (2, 3)]

        mc_before = mc.copy()
        mx_before = mx.copy()

        mc_out, mx_out = epoche(mc, mx, paires, taux_apprentissage=0.0)

        assert np.allclose(mc_out, mc_before)
        assert np.allclose(mx_out, mx_before)

    def test_loss_decreases_over_epoch(self):
        """Après une époque, la perte totale doit baisser (ou au pire stagner légèrement)."""
        C, V, d = 4, 7, 5
        rng = np.random.default_rng(123)
        mc = rng.standard_normal((C, d))
        mx = rng.standard_normal((V, d))
        paires = [(0, 1), (1, 2), (2, 3), (3, 4)]

        loss_before = loss_total(mc, mx, paires)
        mc_out, mx_out = epoche(mc, mx, paires, taux_apprentissage=0.1)
        loss_after = loss_total(mc_out, mx_out, paires)

        assert loss_after <= loss_before + 1e-9

    def test_deterministic_with_seed(self):
        """Avec une seed fixée, l'ordre de permutation est déterministe → résultat reproductible."""
        C, V, d = 3, 6, 4
        mc1 = np.random.randn(C, d)
        mx1 = np.random.randn(V, d)
        mc2 = mc1.copy()
        mx2 = mx1.copy()
        paires = [(0, 1), (1, 2), (2, 3)]

        np.random.seed(42)
        out1 = epoche(mc1, mx1, paires, taux_apprentissage=0.05)

        np.random.seed(42)
        out2 = epoche(mc2, mx2, paires, taux_apprentissage=0.05)

        assert np.allclose(out1[0], out2[0])
        assert np.allclose(out1[1], out2[1])

    def test_single_pair_matches_manual_step(self):
        """Si une seule paire, epoche doit être équivalente à un seul appel descente_de_gradient."""
        # Prépare un petit cas
        C, V, d = 2, 3, 4
        mc_a = np.random.randn(C, d)
        mx_a = np.random.randn(V, d)
        mc_b = mc_a.copy()
        mx_b = mx_a.copy()
        pair = [(1, 2)]
        eta = 0.05

        # Chemin A : via epoche
        mc_out_a, mx_out_a = epoche(mc_a, mx_a, pair, taux_apprentissage=eta)

        # Chemin B : manuel (une étape)
        id_c, id_pos = pair[0]
        p = softmax_scores(mx_b, mc_b[id_c])
        # on suppose la même signature que précédemment
        mc_b = descente_de_gradient(mc_b, mx_b, p, id_c, id_pos, eta)
        # descente_de_gradient met aussi à jour mx_b en place

        assert np.allclose(mc_out_a, mc_b)
        assert np.allclose(mx_out_a, mx_b)

    def test_dimension_mismatch_raises(self):
        """Si les embeddings n'ont pas la même dimension, epoche doit échouer (matmul)."""
        C, V = 2, 3
        mc = np.random.randn(C, 5)   # d=5
        mx = np.random.randn(V, 4)   # d=4 incompatible
        paires = [(0, 1)]

        with pytest.raises(ValueError):
            _ = epoche(mc, mx, paires, taux_apprentissage=0.1)

    def test_inplace_objects(self):
        """Vérifie que les objets retournés sont les mêmes références (MAJ en place)."""
        C, V, d = 3, 4, 2
        mc = np.random.randn(C, d)
        mx = np.random.randn(V, d)
        paires = [(0, 1), (1, 2)]

        mc_id_before = id(mc)
        mx_id_before = id(mx)

        mc_out, mx_out = epoche(mc, mx, paires, taux_apprentissage=0.05)

        assert id(mc_out) == mc_id_before
        assert id(mx_out) == mx_id_before


# Helpers de référence (pour vérifier indépendamment tes fonctions)
def softmax_scores(M, v_c):
    s = M @ v_c
    e = np.exp(s - np.max(s))
    return e / np.sum(e)

def perte_tot_ref(matrice_centre, matrice_contexte, paires):
    total = 0.0
    for (id_c, id_pos) in paires:
        v_c = matrice_centre[id_c]
        p = softmax_scores(matrice_contexte, v_c)
        total += -np.log(p[id_pos])
    return float(total)

class TestPipeline:
    def test_softmax_basic(self):
        C, V, d = 3, 4, 5
        mc = np.random.randn(C, d)
        mx = np.random.randn(V, d)
        p = softmax(mc, mx, id_mot_centre=1)
        assert p.shape == (V,)
        assert np.all(p >= 0)
        assert np.isclose(np.sum(p), 1.0, atol=1e-8)

    def test_descente_updates_center_and_context(self):
        C, V, d = 3, 5, 4
        rng = np.random.default_rng(0)
        mc = rng.standard_normal((C, d))
        mx = rng.standard_normal((V, d))

        id_c = 1
        id_pos = 3
        eta = 0.05

        v_c_before = mc[id_c].copy()
        p = softmax(mc, mx, id_c)

        # copies pour évaluer le delta attendu sur M
        M_before = mx.copy()

        # On calcule l'update attendu sur M avec v_c AVANT MAJ centre :
        y = np.zeros(V, dtype=float); y[id_pos] = 1.0
        expected_delta_M = -eta * np.outer((p - y), v_c_before)

        # appel de ta fonction
        mc_after = descente_de_gradient(mc, mx, p, id_c, id_pos, eta)

        # vérifications
        assert not np.allclose(mc_after[id_c], v_c_before)            # v_c a changé
        assert np.allclose(mx - M_before, expected_delta_M, atol=1e-8) # M a changé comme attendu

    def test_epoche_shapes_and_effect(self):
        C, V, d = 4, 6, 3
        rng = np.random.default_rng(1)
        mc = rng.standard_normal((C, d))
        mx = rng.standard_normal((V, d))
        paires = [(0, 1), (1, 2), (2, 3), (3, 4)]

        mc0, mx0 = mc.copy(), mx.copy()
        eta = 0.05
        mc_out, mx_out = epoche(mc, mx, paires, eta)

        assert mc_out.shape == mc0.shape
        assert mx_out.shape == mx0.shape
        # On s'attend à un changement (sauf hasard extrême)
        assert not np.allclose(mc_out, mc0)
        assert not np.allclose(mx_out, mx0)

    def test_perte_tot_matches_reference(self):
        C, V, d = 3, 5, 4
        rng = np.random.default_rng(2)
        mc = rng.standard_normal((C, d))
        mx = rng.standard_normal((V, d))
        paires = [(0, 1), (1, 2), (2, 3)]

        # ta perte_tot DOIT maintenant appeler softmax(mc, mx, id_c)
        L = perte_tot(mc, mx, paires)
        L_ref = perte_tot_ref(mc, mx, paires)
        assert np.isfinite(L)
        assert np.isclose(L, L_ref, rtol=1e-6, atol=1e-8)

    def test_entrainer_zero_lr_no_change(self):
        C, V, d = 3, 5, 4
        mc = np.random.randn(C, d)
        mx = np.random.randn(V, d)
        paires = [(0, 1), (1, 2)]
        mc0, mx0 = mc.copy(), mx.copy()

        mc2, mx2 = entrainer(mc, mx, paires, taux_apprentissage=0.0, n_epochs=5)
        assert np.allclose(mc2, mc0)
        assert np.allclose(mx2, mx0)

    def test_entrainer_loss_decreases_or_stagnates(self):
        C, V, d = 4, 7, 5
        rng = np.random.default_rng(3)
        mc = rng.standard_normal((C, d))
        mx = rng.standard_normal((V, d))
        paires = [(0, 1), (1, 2), (2, 3), (3, 4)]

        L0 = perte_tot_ref(mc, mx, paires)
        mc2, mx2 = entrainer(mc, mx, paires, taux_apprentissage=0.1, n_epochs=6, patience=3, tol=1e-4, decay=0.95)
        L1 = perte_tot_ref(mc2, mx2, paires)

        assert L1 <= L0 + 1e-9

    def test_entrainer_early_stopping_equiv_one_epoch(self):
        """Avec patience=0 et tol énorme, on force l'arrêt après 1 époque : même résultat que epoche seule (seed fixée)."""
        C, V, d = 3, 6, 4
        mcA = np.random.randn(C, d)
        mxA = np.random.randn(V, d)
        mcB = mcA.copy(); mxB = mxA.copy()
        paires = [(0, 1), (1, 2), (2, 3)]
        eta = 0.05

        np.random.seed(42)
        outA = entrainer(mcA, mxA, paires, taux_apprentissage=eta, n_epochs=100, patience=0, tol=1e9, decay=0.95)

        np.random.seed(42)
        outB = epoche(mcB, mxB, paires, eta)

        assert np.allclose(outA[0], outB[0])
        assert np.allclose(outA[1], outB[1])

    def test_dimension_mismatch_raises(self):
        """Si les dimensions d'embedding ne correspondent pas, les produits matriels doivent échouer."""
        C, V = 2, 3
        mc = np.random.randn(C, 5)   # d=5
        mx = np.random.randn(V, 4)   # d=4 incompatible
        paires = [(0, 1)]
        with pytest.raises(ValueError):
            _ = entrainer(mc, mx, paires, taux_apprentissage=0.05, n_epochs=1)





# Corpus jouet (sans accents)
TEXTE_KQ = (
    "le roi aime la reine le roi porte la couronne "
    "la reine aime le roi la reine porte la couronne"
)

def build_vocab_pairs(texte, window):
    tokens = [tokenisation(texte)]
    freq = frequence_mots(tokens)
    vocab, stoi, itos = vocabulaire(freq)
    paires = creation_paires(tokens, stoi, taille_fenetre=window)
    return tokens, freq, vocab, stoi, itos, paires


class TestYourPipelineOnly:
    def test_softmax_properties(self):
        """softmax: forme (V,), non-négatif, somme=1"""
        C, V, d = 3, 5, 4
        mc = np.random.randn(C, d)
        mx = np.random.randn(V, d)
        p = softmax(mc, mx, id_mot_centre=1)
        assert p.shape == (V,)
        assert np.all(p >= 0)
        assert np.isclose(np.sum(p), 1.0, atol=1e-8)

    def test_descente_updates_center_and_context(self):
        """descente_de_gradient: MAJ de v_c et M conforme à -eta (p - y) v_c_avant^T"""
        C, V, d = 3, 5, 4
        rng = np.random.default_rng(0)
        mc = rng.standard_normal((C, d))
        mx = rng.standard_normal((V, d))

        id_c = 1
        id_pos = 3
        eta = 0.05

        v_c_before = mc[id_c].copy()
        p = softmax(mc, mx, id_c)
        y = np.zeros(V, dtype=float); y[id_pos] = 1.0
        expected_delta_M = -eta * np.outer((p - y), v_c_before)
        M_before = mx.copy()

        mc_after = descente_de_gradient(mc, mx, p, id_c, id_pos, eta)

        assert not np.allclose(mc_after[id_c], v_c_before)
        assert np.allclose(mx - M_before, expected_delta_M, atol=1e-8)

    def test_epoche_changes_params(self):
        """epoche: fait évoluer au moins une des matrices (avec seed fixée)."""
        _, _, vocab, stoi, _, paires = build_vocab_pairs(TEXTE_KQ, window=1)
        V, d = len(vocab), 8
        rng = np.random.default_rng(1)
        mc = rng.standard_normal((V, d))
        mx = rng.standard_normal((V, d))
        mc0, mx0 = mc.copy(), mx.copy()

        eta = 0.05
        mc_out, mx_out = epoche(mc, mx, paires, eta)

        assert mc_out.shape == (V, d) and mx_out.shape == (V, d)
        assert not np.allclose(mc_out, mc0) or not np.allclose(mx_out, mx0)

    def test_perte_tot_uses_your_softmax(self):
        """perte_tot: cohérente avec une recomposition manuelle basée sur TON softmax."""
        _, _, vocab, stoi, _, paires = build_vocab_pairs(TEXTE_KQ, window=1)
        V, d = len(vocab), 10
        mc = np.random.randn(V, d)
        mx = np.random.randn(V, d)

        # ta perte_tot
        L = perte_tot(mc, mx, paires)

        # recomposition "manuelle" mais avec TON softmax
        total = 0.0
        for (id_c, id_pos) in paires:
            p = softmax(mc, mx, id_c)  # <— on appelle bien TA softmax
            total += -np.log(p[id_pos])
        assert np.isfinite(L)
        assert np.isclose(L, float(total), rtol=1e-9, atol=1e-12)

    def test_entrainer_zero_lr_no_change(self):
        """entrainer avec eta=0 ne modifie pas les matrices."""
        _, _, vocab, stoi, _, paires = build_vocab_pairs(TEXTE_KQ, window=1)
        V, d = len(vocab), 9
        mc = np.random.randn(V, d); mx = np.random.randn(V, d)
        mc0, mx0 = mc.copy(), mx.copy()

        mc2, mx2 = entrainer(mc, mx, paires, taux_apprentissage=0.0, n_epochs=3)
        assert np.allclose(mc2, mc0)
        assert np.allclose(mx2, mx0)

    def test_entrainer_loss_decreases_with_more_epochs(self):
        """Même init : plus d'époques => perte totale plus faible (ou égale)."""
        dimension = 12
        np.random.seed(123)
        mc1, mx1 = deep_learning(
            TEXTE_KQ, dimension=dimension, taux_apprentissage=0.05,
            window_size=1, n_epochs=1, patience=3, tolerance=1e-4, decay=0.95
        )
        np.random.seed(123)
        mc5, mx5 = deep_learning(
            TEXTE_KQ, dimension=dimension, taux_apprentissage=0.05,
            window_size=1, n_epochs=5, patience=3, tolerance=1e-4, decay=0.95
        )

        tokens = [tokenisation(TEXTE_KQ)]
        _, _, vocab, stoi, _, _ = build_vocab_pairs(TEXTE_KQ, window=1)
        paires = creation_paires(tokens, stoi, 1)

        L1 = perte_tot(mc1, mx1, paires)
        L5 = perte_tot(mc5, mx5, paires)
        assert L5 <= L1 + 1e-9

    def test_deep_learning_shapes_and_empty_text(self):
        """deep_learning renvoie des matrices de bonne taille, et gère le texte vide."""
        mc, mx = deep_learning(TEXTE_KQ, dimension=7, taux_apprentissage=0.03,
                               window_size=2, n_epochs=2, patience=2, tolerance=1e-3, decay=0.95)
        assert mc.ndim == 2 and mx.ndim == 2
        assert mc.shape[0] == mx.shape[0] and mc.shape[1] == mx.shape[1]

        mc0, mx0 = deep_learning("", dimension=6, taux_apprentissage=0.03,
                                 window_size=1, n_epochs=1, patience=1, tolerance=1e-3, decay=0.95)
        assert mc0.shape == (0, 6)
        assert mx0.shape == (0, 6)

    def test_dimension_mismatch_raises_in_epoche(self):
        """Incompatibilité d'embedding: les produits dans epoche doivent échouer."""
        V, d1, d2 = 5, 6, 7
        mc = np.random.randn(V, d1)
        mx = np.random.randn(V, d2)  # d2 != d1
        # on fabrique des paires triviales sur des IDs existants
        paires = [(i, (i+1) % V) for i in range(V)]
        with pytest.raises(ValueError):
            _ = epoche(mc, mx, paires, taux_apprentissage=0.05)

# Petit corpus jouet (sans accents)
TEXTE_KQ = (
    "le roi aime la reine le roi porte la couronne "
    "la reine aime le roi la reine porte la couronne"
)

def test_deep_learning_shapes_match_vocab():
    """deep_learning renvoie (|V|, d) pour centre et contexte."""
    dimension = 8
    np.random.seed(0)
    mc, mx = deep_learning(
        TEXTE_KQ, dimension=dimension, taux_apprentissage=0.05,
        window_size=1, n_epochs=3, patience=3, tolerance=1e-3, decay=0.95
    )

    # Calcule |V| via TES fonctions
    tokens = [tokenisation(TEXTE_KQ)]
    freq = frequence_mots(tokens)
    vocab, stoi, itos = vocabulaire(freq)
    V = len(vocab)

    assert mc.shape == (V, dimension)
    assert mx.shape == (V, dimension)

def test_deep_learning_empty_text():
    """Texte vide -> matrices vides (0, d)."""
    dimension = 6
    np.random.seed(1)
    mc, mx = deep_learning(
        "", dimension=dimension, taux_apprentissage=0.05,
        window_size=1, n_epochs=2, patience=2, tolerance=1e-3, decay=0.95
    )
    assert mc.shape == (0, dimension)
    assert mx.shape == (0, dimension)

def test_zero_learning_rate_reproduces_initialisation():
    """
    Avec eta=0, aucune MAJ ne doit avoir lieu.
    On vérifie que la sortie == matrices d'init générées avec la même seed.
    (On n'utilise que TES fonctions pour |V| et paires.)
    """
    dimension = 9
    tokens = [tokenisation(TEXTE_KQ)]
    freq = frequence_mots(tokens)
    vocab, stoi, itos = vocabulaire(freq)
    V = len(vocab)

    # Reconstruire l'init EXACTE (rand - 0.5) avec la même seed que l'appel à deep_learning
    np.random.seed(2024)
    mc_init = np.random.rand(V, dimension) - 0.5
    mx_init = np.random.rand(V, dimension) - 0.5

    np.random.seed(2024)
    mc_out, mx_out = deep_learning(
        TEXTE_KQ, dimension=dimension, taux_apprentissage=0.0,
        window_size=1, n_epochs=3, patience=3, tolerance=1e-3, decay=0.95
    )

    assert np.allclose(mc_out, mc_init)
    assert np.allclose(mx_out, mx_init)

def test_loss_decreases_with_more_epochs():
    """Même initialisation (seed identique) : plus d'époques => perte totale plus faible (ou égale)."""
    dimension = 12

    # même seed pour avoir la même init
    np.random.seed(123)
    mc1, mx1 = deep_learning(
        TEXTE_KQ, dimension=dimension, taux_apprentissage=0.05,
        window_size=1, n_epochs=1, patience=3, tolerance=1e-4, decay=0.95
    )

    np.random.seed(123)
    mc5, mx5 = deep_learning(
        TEXTE_KQ, dimension=dimension, taux_apprentissage=0.05,
        window_size=1, n_epochs=5, patience=3, tolerance=1e-4, decay=0.95
    )

    # Perte via TES fonctions
    tokens = [tokenisation(TEXTE_KQ)]
    freq = frequence_mots(tokens)
    vocab, stoi, itos = vocabulaire(freq)
    paires = creation_paires(tokens, stoi, 1)

    L1 = perte_tot(mc1, mx1, paires)
    L5 = perte_tot(mc5, mx5, paires)
    assert np.isfinite(L1) and np.isfinite(L5)
    assert L5 <= L1 + 1e-9

def test_reproducibility_with_seed():
    """Même seed ⇒ mêmes matrices (reproductibilité)."""
    dimension = 10
    np.random.seed(42)
    mc_a, mx_a = deep_learning(
        TEXTE_KQ, dimension=dimension, taux_apprentissage=0.03,
        window_size=2, n_epochs=4, patience=3, tolerance=1e-3, decay=0.95
    )
    np.random.seed(42)
    mc_b, mx_b = deep_learning(
        TEXTE_KQ, dimension=dimension, taux_apprentissage=0.03,
        window_size=2, n_epochs=4, patience=3, tolerance=1e-3, decay=0.95
    )
    assert np.allclose(mc_a, mc_b)
    assert np.allclose(mx_a, mx_b)

def test_window_size_changes_training():
    """Même seed, fenêtres différentes ⇒ résultats (très probablement) différents."""
    dimension = 8
    np.random.seed(7)
    mc_w1, mx_w1 = deep_learning(
        TEXTE_KQ, dimension=dimension, taux_apprentissage=0.05,
        window_size=1, n_epochs=3, patience=3, tolerance=1e-3, decay=0.95
    )
    np.random.seed(7)
    mc_w2, mx_w2 = deep_learning(
        TEXTE_KQ, dimension=dimension, taux_apprentissage=0.05,
        window_size=2, n_epochs=3, patience=3, tolerance=1e-3, decay=0.95
    )
    # Il est attendu que l'un des deux couples diffère
    assert not np.allclose(mc_w1, mc_w2) or not np.allclose(mx_w1, mx_w2)

def test_no_nan_no_inf_after_training():
    """Après deep_learning, la perte doit être finie (pas NaN/Inf)."""
    dimension = 11
    np.random.seed(9)
    mc, mx = deep_learning(
        TEXTE_KQ, dimension=dimension, taux_apprentissage=0.04,
        window_size=2, n_epochs=2, patience=2, tolerance=1e-3, decay=0.95
    )
    tokens = [tokenisation(TEXTE_KQ)]
    freq = frequence_mots(tokens)
    vocab, stoi, itos = vocabulaire(freq)
    paires = creation_paires(tokens, stoi, 2)
    L = perte_tot(mc, mx, paires)
    assert np.isfinite(L)
