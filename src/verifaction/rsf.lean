variable {F : Type}

def succ_not_zero (n : Nat) (h : Nat.succ n = 0) : False :=
  nomatch h

def zero_not_succ (n : Nat) (h : 0 = Nat.succ n) : False :=
  nomatch h

def succ_inj {n m : Nat} (h : Nat.succ n = Nat.succ m) : n = m :=
  congrArg (fun x => match x with | 0 => 0 | Nat.succ k => k) h

def nat_zero_add : (m : Nat) → 0 + m = m
| 0 => Eq.refl 0
| Nat.succ m => congrArg Nat.succ (nat_zero_add m)

def nat_succ_add (n : Nat) : (m : Nat) → Nat.succ n + m = Nat.succ (n + m)
| 0 => Eq.refl (Nat.succ n)
| Nat.succ m => congrArg Nat.succ (nat_succ_add n m)

def congr_cons {α : Type} {x y : α} {xs ys : List α} (hx : x = y) (hxs : xs = ys) : x :: xs = y :: ys :=
  match hx, hxs with
  | Eq.refl _, Eq.refl _ => Eq.refl _

def congr_append {α : Type} {xs ys us vs : List α} (h1 : xs = ys) (h2 : us = vs) : xs ++ us = ys ++ vs :=
  match h1, h2 with
  | Eq.refl _, Eq.refl _ => Eq.refl _

def congr_add {n1 n2 m1 m2 : Nat} (h1 : n1 = n2) (h2 : m1 = m2) : n1 + m1 = n2 + m2 :=
  match h1, h2 with
  | Eq.refl _, Eq.refl _ => Eq.refl _

def split_at (n : Nat) (L : List F) : List F × List F :=
  match n with
  | 0 => ([], L)
  | Nat.succ k =>
    match L with
    | [] => ([], [])
    | x :: xs =>
      let (l1, l2) := split_at k xs
      (x :: l1, l2)

theorem split_at_lengths_gen : (n m : Nat) → (L : List F) → L.length = n + m →
  (split_at n L).1.length = n ∧ (split_at n L).2.length = m
| 0, m, L, h =>
  ⟨Eq.refl 0, Eq.trans h (nat_zero_add m)⟩
| Nat.succ n, m, [], h =>
  False.elim (zero_not_succ (n + m) (Eq.trans h (nat_succ_add n m)))
| Nat.succ n, m, _ :: xs, h =>
  let h' : Nat.succ xs.length = Nat.succ (n + m) := Eq.trans h (nat_succ_add n m)
  let h_len : xs.length = n + m := succ_inj h'
  let rec_res := split_at_lengths_gen n m xs h_len
  ⟨congrArg Nat.succ rec_res.left, rec_res.right⟩

theorem split_at_reconstruct : (n : Nat) → (L : List F) →
  (split_at n L).1 ++ (split_at n L).2 = L
| 0, L => Eq.refl L
| Nat.succ _, [] => Eq.refl []
| Nat.succ n, x :: xs =>
  let rec_res := split_at_reconstruct n xs
  congrArg (List.cons x) rec_res

theorem split_at_append : (n : Nat) → (L1 L2 : List F) → L1.length = n →
  split_at n (L1 ++ L2) = (L1, L2)
| 0, [], L2, _ => Eq.refl ([], L2)
| 0, _ :: _, _, h => False.elim (succ_not_zero _ h)
| Nat.succ n, x :: xs, L2, h =>
  let h_rec := split_at_append n xs L2 (succ_inj h)
  congrArg (fun p => (x :: p.1, p.2)) h_rec
| Nat.succ _, [], _, h => False.elim (zero_not_succ _ h)

theorem length_zipWith : (f : F → F → F) → (L1 L2 : List F) → L1.length = L2.length →
  (List.zipWith f L1 L2).length = L1.length
| _, [], [], _ => Eq.refl 0
| f, _ :: xs, _ :: ys, h =>
  let h_len : xs.length = ys.length := succ_inj h
  congrArg Nat.succ (length_zipWith f xs ys h_len)
| _, _ :: _, [], h => False.elim (succ_not_zero _ h)
| _, [], _ :: _, h => False.elim (zero_not_succ _ h)

theorem length_append : (L1 L2 : List F) → (L1 ++ L2).length = L1.length + L2.length
| [], L2 => Eq.symm (nat_zero_add L2.length)
| _ :: xs, L2 =>
  Eq.trans (congrArg Nat.succ (length_append xs L2)) (Eq.symm (nat_succ_add xs.length L2.length))

structure RsfAlgebra (F : Type) where
  add : F → F → F
  sub : F → F → F
  mul : F → F → F
  div : F → F → F
  S : List F → List F
  T : List F → List F
  add_sub_cancel : ∀ a b : F, sub (add a b) b = a
  sub_add_cancel : ∀ a b : F, add (sub a b) b = a
  mul_div_cancel : ∀ a b : F, div (mul a b) b = a
  div_mul_cancel : ∀ a b : F, mul (div a b) b = a
  S_len : ∀ L, (S L).length = L.length
  T_len : ∀ L, (T L).length = L.length

theorem zipWith_add_sub_cancel (ra : RsfAlgebra F) : (L1 L2 : List F) →
  List.zipWith ra.sub (List.zipWith ra.add L1 L2) L2 = L1
| [], [] => Eq.refl []
| [], _::_ => Eq.refl []
| _::_, [] => Eq.refl []
| x::xs, y::ys =>
  let rec_res := zipWith_add_sub_cancel ra xs ys
  let step : ra.sub (ra.add x y) y = x := ra.add_sub_cancel x y
  congr_cons step rec_res

theorem zipWith_mul_div_cancel (ra : RsfAlgebra F) : (L1 L2 : List F) →
  List.zipWith ra.div (List.zipWith ra.mul L1 L2) L2 = L1
| [], [] => Eq.refl []
| [], _::_ => Eq.refl []
| _::_, [] => Eq.refl []
| x::xs, y::ys =>
  let rec_res := zipWith_mul_div_cancel ra xs ys
  let step : ra.div (ra.mul x y) y = x := ra.mul_div_cancel x y
  congr_cons step rec_res

theorem zipWith_sub_add_cancel (ra : RsfAlgebra F) : (L1 L2 : List F) →
  List.zipWith ra.add (List.zipWith ra.sub L1 L2) L2 = L1
| [], [] => Eq.refl []
| [], _::_ => Eq.refl []
| _::_, [] => Eq.refl []
| x::xs, y::ys =>
  let rec_res := zipWith_sub_add_cancel ra xs ys
  let step : ra.add (ra.sub x y) y = x := ra.sub_add_cancel x y
  congr_cons step rec_res

theorem zipWith_div_mul_cancel (ra : RsfAlgebra F) : (L1 L2 : List F) →
  List.zipWith ra.mul (List.zipWith ra.div L1 L2) L2 = L1
| [], [] => Eq.refl []
| [], _::_ => Eq.refl []
| _::_, [] => Eq.refl []
| x::xs, y::ys =>
  let rec_res := zipWith_div_mul_cancel ra xs ys
  let step : ra.mul (ra.div x y) y = x := ra.div_mul_cancel x y
  congr_cons step rec_res

def forwardCore (ra : RsfAlgebra F) (dim : Nat) (L : List F) : List F :=
  let x1 := (split_at dim L).1
  let x2 := (split_at dim L).2
  let scale := ra.S x2
  let x1_new := List.zipWith ra.mul x1 scale
  let trans := ra.T x1_new
  let x2_new := List.zipWith ra.add x2 trans
  x1_new ++ x2_new

def inverseCore (ra : RsfAlgebra F) (dim : Nat) (L : List F) : List F :=
  let y1 := (split_at dim L).1
  let y2 := (split_at dim L).2
  let trans := ra.T y1
  let y2_new := List.zipWith ra.sub y2 trans
  let scale := ra.S y2_new
  let y1_new := List.zipWith ra.div y1 scale
  y1_new ++ y2_new

theorem rsf_invertible (ra : RsfAlgebra F) (dim : Nat) (L : List F) (h : L.length = dim + dim) :
  inverseCore ra dim (forwardCore ra dim L) = L :=
  let x1 := (split_at dim L).1
  let x2 := (split_at dim L).2
  let h_splits : x1.length = dim ∧ x2.length = dim := split_at_lengths_gen dim dim L h
  let scale := ra.S x2
  let x1_new := List.zipWith ra.mul x1 scale
  let trans := ra.T x1_new
  let x2_new := List.zipWith ra.add x2 trans
  
  let h_scale_len : scale.length = dim := Eq.trans (ra.S_len x2) h_splits.right
  let h_x1_new_len : x1_new.length = dim :=
    Eq.trans (length_zipWith ra.mul x1 scale (Eq.trans h_splits.left (Eq.symm h_scale_len))) h_splits.left
    
  let h_trans_len : trans.length = dim := Eq.trans (ra.T_len x1_new) h_x1_new_len
  let h_x2_new_len : x2_new.length = dim :=
    Eq.trans (length_zipWith ra.add x2 trans (Eq.trans h_splits.right (Eq.symm h_trans_len))) h_splits.right
    
  let h_split_new : split_at dim (x1_new ++ x2_new) = (x1_new, x2_new) :=
    split_at_append dim x1_new x2_new h_x1_new_len

  let step1 : inverseCore ra dim (forwardCore ra dim L) =
    let y1 := (split_at dim (x1_new ++ x2_new)).1;
    let y2 := (split_at dim (x1_new ++ x2_new)).2;
    let trans_inv := ra.T y1;
    let y2_new := List.zipWith ra.sub y2 trans_inv;
    let scale_inv := ra.S y2_new;
    let y1_new := List.zipWith ra.div y1 scale_inv;
    y1_new ++ y2_new := Eq.refl _
    
  let step2 : (let y1 := (split_at dim (x1_new ++ x2_new)).1;
    let y2 := (split_at dim (x1_new ++ x2_new)).2;
    let trans_inv := ra.T y1;
    let y2_new := List.zipWith ra.sub y2 trans_inv;
    let scale_inv := ra.S y2_new;
    let y1_new := List.zipWith ra.div y1 scale_inv;
    y1_new ++ y2_new) =
    (let trans_inv := ra.T x1_new;
    let y2_new := List.zipWith ra.sub x2_new trans_inv;
    let scale_inv := ra.S y2_new;
    let y1_new := List.zipWith ra.div x1_new scale_inv;
    y1_new ++ y2_new) :=
    congrArg (fun p =>
      let y1 := p.1; let y2 := p.2;
      let trans_inv := ra.T y1;
      let y2_new := List.zipWith ra.sub y2 trans_inv;
      let scale_inv := ra.S y2_new;
      let y1_new := List.zipWith ra.div y1 scale_inv;
      y1_new ++ y2_new) h_split_new

  let trans_inv := ra.T x1_new
  let h_trans_inv : trans_inv = trans := Eq.refl trans_inv

  let y2_new := List.zipWith ra.sub x2_new trans_inv
  let h_y2_new : y2_new = x2 :=
    let h_subst : y2_new = List.zipWith ra.sub (List.zipWith ra.add x2 trans) trans :=
      congrArg (fun t => List.zipWith ra.sub x2_new t) h_trans_inv
    let h_cancel : List.zipWith ra.sub (List.zipWith ra.add x2 trans) trans = x2 :=
      zipWith_add_sub_cancel ra x2 trans
    Eq.trans h_subst h_cancel

  let scale_inv := ra.S y2_new
  let h_scale_inv : scale_inv = scale :=
    congrArg ra.S h_y2_new

  let y1_new := List.zipWith ra.div x1_new scale_inv
  let h_y1_new : y1_new = x1 :=
    let h_subst : y1_new = List.zipWith ra.div x1_new scale :=
      congrArg (fun s => List.zipWith ra.div x1_new s) h_scale_inv
    let h_cancel : List.zipWith ra.div x1_new scale = List.zipWith ra.div (List.zipWith ra.mul x1 scale) scale :=
      Eq.refl _
    let h_cancel2 : List.zipWith ra.div (List.zipWith ra.mul x1 scale) scale = x1 :=
      zipWith_mul_div_cancel ra x1 scale
    Eq.trans h_subst (Eq.trans h_cancel h_cancel2)

  let step3 : (let trans_inv := ra.T x1_new;
    let y2_new := List.zipWith ra.sub x2_new trans_inv;
    let scale_inv := ra.S y2_new;
    let y1_new := List.zipWith ra.div x1_new scale_inv;
    y1_new ++ y2_new) = x1 ++ x2 :=
    congr_append h_y1_new h_y2_new

  let step4 : x1 ++ x2 = L := split_at_reconstruct dim L

  Eq.trans step1 (Eq.trans step2 (Eq.trans step3 step4))

theorem rsf_invertible_rev (ra : RsfAlgebra F) (dim : Nat) (L : List F) (h : L.length = dim + dim) :
  forwardCore ra dim (inverseCore ra dim L) = L :=
  let y1 := (split_at dim L).1
  let y2 := (split_at dim L).2
  let h_splits : y1.length = dim ∧ y2.length = dim := split_at_lengths_gen dim dim L h
  let trans := ra.T y1
  let y2_new := List.zipWith ra.sub y2 trans
  let scale := ra.S y2_new
  let y1_new := List.zipWith ra.div y1 scale

  let h_trans_len : trans.length = dim := Eq.trans (ra.T_len y1) h_splits.left
  let h_y2_new_len : y2_new.length = dim :=
    Eq.trans (length_zipWith ra.sub y2 trans (Eq.trans h_splits.right (Eq.symm h_trans_len))) h_splits.right
  let h_scale_len : scale.length = dim := Eq.trans (ra.S_len y2_new) h_y2_new_len
  let h_y1_new_len : y1_new.length = dim :=
    Eq.trans (length_zipWith ra.div y1 scale (Eq.trans h_splits.left (Eq.symm h_scale_len))) h_splits.left

  let h_split_new : split_at dim (y1_new ++ y2_new) = (y1_new, y2_new) :=
    split_at_append dim y1_new y2_new h_y1_new_len

  let step1 : forwardCore ra dim (inverseCore ra dim L) =
    let x1 := (split_at dim (y1_new ++ y2_new)).1;
    let x2 := (split_at dim (y1_new ++ y2_new)).2;
    let scale_fwd := ra.S x2;
    let x1_new_res := List.zipWith ra.mul x1 scale_fwd;
    let trans_fwd := ra.T x1_new_res;
    let x2_new_res := List.zipWith ra.add x2 trans_fwd;
    x1_new_res ++ x2_new_res := Eq.refl _

  let step2 : (let x1 := (split_at dim (y1_new ++ y2_new)).1;
    let x2 := (split_at dim (y1_new ++ y2_new)).2;
    let scale_fwd := ra.S x2;
    let x1_new_res := List.zipWith ra.mul x1 scale_fwd;
    let trans_fwd := ra.T x1_new_res;
    let x2_new_res := List.zipWith ra.add x2 trans_fwd;
    x1_new_res ++ x2_new_res) =
    (let scale_fwd := ra.S y2_new;
    let x1_new_res := List.zipWith ra.mul y1_new scale_fwd;
    let trans_fwd := ra.T x1_new_res;
    let x2_new_res := List.zipWith ra.add y2_new trans_fwd;
    x1_new_res ++ x2_new_res) :=
    congrArg (fun p =>
      let x1 := p.1; let x2 := p.2;
      let scale_fwd := ra.S x2;
      let x1_new_res := List.zipWith ra.mul x1 scale_fwd;
      let trans_fwd := ra.T x1_new_res;
      let x2_new_res := List.zipWith ra.add x2 trans_fwd;
      x1_new_res ++ x2_new_res) h_split_new

  let scale_fwd := ra.S y2_new
  let h_scale_fwd : scale_fwd = scale := Eq.refl scale_fwd

  let x1_new_res := List.zipWith ra.mul y1_new scale_fwd
  let h_x1_new_res : x1_new_res = y1 :=
    let h_subst : x1_new_res = List.zipWith ra.mul y1_new scale :=
      congrArg (fun s => List.zipWith ra.mul y1_new s) h_scale_fwd
    let h_cancel : List.zipWith ra.mul y1_new scale = List.zipWith ra.mul (List.zipWith ra.div y1 scale) scale :=
      Eq.refl _
    let h_cancel2 : List.zipWith ra.mul (List.zipWith ra.div y1 scale) scale = y1 :=
      zipWith_div_mul_cancel ra y1 scale
    Eq.trans h_subst (Eq.trans h_cancel h_cancel2)

  let trans_fwd := ra.T x1_new_res
  let h_trans_fwd : trans_fwd = trans :=
    congrArg ra.T h_x1_new_res

  let x2_new_res := List.zipWith ra.add y2_new trans_fwd
  let h_x2_new_res : x2_new_res = y2 :=
    let h_subst : x2_new_res = List.zipWith ra.add y2_new trans :=
      congrArg (fun t => List.zipWith ra.add y2_new t) h_trans_fwd
    let h_cancel : List.zipWith ra.add y2_new trans = List.zipWith ra.add (List.zipWith ra.sub y2 trans) trans :=
      Eq.refl _
    let h_cancel2 : List.zipWith ra.add (List.zipWith ra.sub y2 trans) trans = y2 :=
      zipWith_sub_add_cancel ra y2 trans
    Eq.trans h_subst (Eq.trans h_cancel h_cancel2)

  let step3 : (let scale_fwd := ra.S y2_new;
    let x1_new_res := List.zipWith ra.mul y1_new scale_fwd;
    let trans_fwd := ra.T x1_new_res;
    let x2_new_res := List.zipWith ra.add y2_new trans_fwd;
    x1_new_res ++ x2_new_res) = y1 ++ y2 :=
    congr_append h_x1_new_res h_x2_new_res

  let step4 : y1 ++ y2 = L := split_at_reconstruct dim L

  Eq.trans step1 (Eq.trans step2 (Eq.trans step3 step4))

def ValidShape : List Nat → Prop
| [] => False
| [x] => x > 0
| x :: xs => x > 0 ∧ ValidShape xs

def shapeProd : List Nat → Nat
| [] => 1
| x :: xs => x * shapeProd xs

structure Tensor (F : Type) where
  shape : List Nat
  data : List F
  h_valid : ValidShape shape
  h_len : data.length = shapeProd shape

def init_tensor_spec (F : Type) (shape : List Nat) (data : List F) : Prop :=
  ValidShape shape ∧ data.length = shapeProd shape

theorem tensor_init_rejects_invalid_shape (shape : List Nat) (h_invalid : ¬ ValidShape shape) :
  (d : List F) → ¬ init_tensor_spec F shape d :=
  fun _ h_spec => h_invalid h_spec.left

theorem valid_shape_empty_is_false : ¬ ValidShape [] :=
  fun h => h

theorem valid_shape_zero_is_false : (xs : List Nat) → ¬ ValidShape (0 :: xs)
| [] => fun h => Nat.lt_irrefl 0 h
| _ :: _ => fun h => Nat.lt_irrefl 0 h.left

def usize_max : Nat := 18446744073709551615

def ValidDim (dim : Nat) : Prop :=
  dim > 0 ∧ dim ≤ usize_max / 2

def ValidLen (dim : Nat) (n : Nat) : Prop :=
  n = dim + dim

def transform_precondition (dim : Nat) (L : List F) : Prop :=
  ValidDim dim ∧ L.length = dim + dim

theorem rsf_transform_mismatched_size (dim : Nat) (L : List F) (h_mismatch : L.length ≠ dim + dim) :
  ¬ transform_precondition dim L :=
  fun h => h_mismatch h.right

def rsfForwardCore (ra : RsfAlgebra F) (dim : Nat) (L : List F) (_ : transform_precondition dim L) : List F :=
  forwardCore ra dim L

def rsfInverseCore (ra : RsfAlgebra F) (dim : Nat) (L : List F) (_ : transform_precondition dim L) : List F :=
  inverseCore ra dim L

theorem length_forwardCore (ra : RsfAlgebra F) (dim : Nat) (L : List F) (h : L.length = dim + dim) :
  (forwardCore ra dim L).length = dim + dim :=
  let x1 := (split_at dim L).1
  let x2 := (split_at dim L).2
  let h_splits : x1.length = dim ∧ x2.length = dim := split_at_lengths_gen dim dim L h
  let scale := ra.S x2
  let h_scale_len : scale.length = dim := Eq.trans (ra.S_len x2) h_splits.right
  let x1_new := List.zipWith ra.mul x1 scale
  let h_x1_new_len : x1_new.length = dim :=
    Eq.trans (length_zipWith ra.mul x1 scale (Eq.trans h_splits.left (Eq.symm h_scale_len))) h_splits.left
  let trans := ra.T x1_new
  let h_trans_len : trans.length = dim := Eq.trans (ra.T_len x1_new) h_x1_new_len
  let x2_new := List.zipWith ra.add x2 trans
  let h_x2_new_len : x2_new.length = dim :=
    Eq.trans (length_zipWith ra.add x2 trans (Eq.trans h_splits.right (Eq.symm h_trans_len))) h_splits.right
  let h_append := length_append x1_new x2_new
  let step1 : (x1_new ++ x2_new).length = x1_new.length + x2_new.length := h_append
  let step2 : x1_new.length + x2_new.length = dim + dim :=
    congr_add h_x1_new_len h_x2_new_len
  Eq.trans step1 step2

theorem length_inverseCore (ra : RsfAlgebra F) (dim : Nat) (L : List F) (h : L.length = dim + dim) :
  (inverseCore ra dim L).length = dim + dim :=
  let y1 := (split_at dim L).1
  let y2 := (split_at dim L).2
  let h_splits : y1.length = dim ∧ y2.length = dim := split_at_lengths_gen dim dim L h
  let trans := ra.T y1
  let h_trans_len : trans.length = dim := Eq.trans (ra.T_len y1) h_splits.left
  let y2_new := List.zipWith ra.sub y2 trans
  let h_y2_new_len : y2_new.length = dim :=
    Eq.trans (length_zipWith ra.sub y2 trans (Eq.trans h_splits.right (Eq.symm h_trans_len))) h_splits.right
  let scale := ra.S y2_new
  let h_scale_len : scale.length = dim := Eq.trans (ra.S_len y2_new) h_y2_new_len
  let y1_new := List.zipWith ra.div y1 scale
  let h_y1_new_len : y1_new.length = dim :=
    Eq.trans (length_zipWith ra.div y1 scale (Eq.trans h_splits.left (Eq.symm h_scale_len))) h_splits.left
  let h_append := length_append y1_new y2_new
  let step1 : (y1_new ++ y2_new).length = y1_new.length + y2_new.length := h_append
  let step2 : y1_new.length + y2_new.length = dim + dim :=
    congr_add h_y1_new_len h_y2_new_len
  Eq.trans step1 step2

theorem rsf_round_trip (ra : RsfAlgebra F) (dim : Nat) (L : List F) (h : transform_precondition dim L) :
  let h_forward_pre : transform_precondition dim (rsfForwardCore ra dim L h) :=
    ⟨h.left, length_forwardCore ra dim L h.right⟩
  rsfInverseCore ra dim (rsfForwardCore ra dim L h) h_forward_pre = L :=
  rsf_invertible ra dim L h.right

theorem rsf_round_trip_rev (ra : RsfAlgebra F) (dim : Nat) (L : List F) (h : transform_precondition dim L) :
  let h_inverse_pre : transform_precondition dim (rsfInverseCore ra dim L h) :=
    ⟨h.left, length_inverseCore ra dim L h.right⟩
  rsfForwardCore ra dim (rsfInverseCore ra dim L h) h_inverse_pre = L :=
  rsf_invertible_rev ra dim L h.right
