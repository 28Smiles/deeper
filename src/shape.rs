use std::cmp::max;

pub trait Shape: Copy {
    type Dims: generic_array::ArrayLength<usize>;
    fn size(&self) -> usize;
    fn dimensions(&self) -> generic_array::GenericArray<usize, Self::Dims>;
    fn strides(&self) -> generic_array::GenericArray<usize, Self::Dims>;
}
pub trait MinSizeShape {
    type Value: typenum::Unsigned;
}
pub trait BroadcastShape<Rhs> {
    type Output;
    fn broadcast(self, rhs: Rhs) -> Self::Output;
}

#[derive(Copy, Clone)]
pub struct Dyn {
    size: usize,
}
impl Dyn {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}
#[derive(Copy, Clone)]
pub struct Cst<Size: typenum::Unsigned> {
    _phantom: std::marker::PhantomData<Size>,
}
impl<Size: typenum::Unsigned> Cst<Size> {
    pub(crate) fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

pub trait Dim: Copy {
    fn size(&self) -> usize;
}

impl Dim for Dyn {
    #[inline(always)]
    fn size(&self) -> usize {
        self.size
    }
}
impl<Size: typenum::Unsigned> Dim for Cst<Size> {
    #[inline(always)]
    fn size(&self) -> usize {
        Size::USIZE
    }
}
impl<
    SizeL: typenum::Unsigned,
    SizeR: typenum::Unsigned,
    SizeO: typenum::Unsigned,
    IsOneL: typenum::Bit,
    IsOneR: typenum::Bit,
    IsOne: typenum::Bit,
    IsEq: typenum::Bit,
    IsEqL: typenum::Bit,
    IsEqR: typenum::Bit,
> BroadcastShape<Cst<SizeR>> for Cst<SizeL>
    where
        // l = 1
        // r = 1
        // l = r
        SizeL: typenum::IsEqual<typenum::U1, Output = IsOneL>,
        SizeR: typenum::IsEqual<typenum::U1, Output = IsOneR>,
        SizeL: typenum::IsEqual<SizeR, Output = IsEq>,
        // |= l = 1 | r = 1 | l = r
        IsOneL: core::ops::BitOr<IsOneR, Output = IsOne>,
        IsOne: core::ops::BitOr<IsEq, Output = typenum::True>,
        // o = l <=> r = 1 || r = l
        // o = r <=> l = 1 || l = r
        IsOneR: core::ops::BitOr<IsEq, Output = IsEqL>, // Both are 1
        IsOneL: core::ops::BitOr<IsEq, Output = IsEqR>, // Both are 1

        SizeL: typenum::Max<
            SizeR,
            Output = SizeO,
        >,
{
    type Output = Cst<SizeO>;
    fn broadcast(self, _: Cst<SizeR>) -> Self::Output {
        Cst::new()
    }
}
impl<
    SizeL: typenum::Unsigned
> BroadcastShape<Dyn> for Cst<SizeL> {
    type Output = Dyn;
    fn broadcast(self, rhs: Dyn) -> Self::Output {
        Dyn { size: max(SizeL::USIZE, rhs.size) }
    }
}
impl<
    SizeR: typenum::Unsigned
> BroadcastShape<Cst<SizeR>> for Dyn {
    type Output = Dyn;
    fn broadcast(self, _: Cst<SizeR>) -> Self::Output {
        Dyn { size: max(SizeR::USIZE, self.size) }
    }
}
impl BroadcastShape<Dyn> for Dyn {
    type Output = Dyn;
    fn broadcast(self, rhs: Dyn) -> Self::Output {
        Dyn { size: max(self.size, rhs.size) }
    }
}
impl MinSizeShape for Dyn {
    type Value = typenum::U1;
}
impl<Size: typenum::Unsigned> MinSizeShape for Cst<Size>
{
    type Value = Size;
}

// Scalars
impl Shape for () {
    type Dims = typenum::U0;
    #[inline(always)]
    fn size(&self) -> usize {
        1
    }
    #[inline(always)]
    fn dimensions(&self) -> generic_array::GenericArray<usize, Self::Dims> {
        generic_array::GenericArray::default()
    }
    #[inline(always)]
    fn strides(&self) -> generic_array::GenericArray<usize, Self::Dims> {
        generic_array::GenericArray::default()
    }
}
impl MinSizeShape for () {
    type Value = typenum::U1;
}
impl<S: Shape> BroadcastShape<S> for () {
    type Output = S;
    #[inline(always)]
    fn broadcast(self, rhs: S) -> Self::Output {
        rhs
    }
}

// Vectors
impl<Size0: typenum::Unsigned> Shape for (Cst<Size0>,) {
    type Dims = typenum::U1;
    #[inline(always)]
    fn size(&self) -> usize {
        Size0::USIZE
    }
    #[inline(always)]
    fn dimensions(&self) -> generic_array::GenericArray<usize, Self::Dims> {
        generic_array::GenericArray::from([Size0::USIZE])
    }
    #[inline(always)]
    fn strides(&self) -> generic_array::GenericArray<usize, Self::Dims> {
        generic_array::GenericArray::from([1])
    }
}
impl Shape for (Dyn,) {
    type Dims = typenum::U1;
    #[inline(always)]
    fn size(&self) -> usize {
        self.0.size
    }
    #[inline(always)]
    fn dimensions(&self) -> generic_array::GenericArray<usize, Self::Dims> {
        generic_array::GenericArray::from([self.0.size])
    }
    #[inline(always)]
    fn strides(&self) -> generic_array::GenericArray<usize, Self::Dims> {
        generic_array::GenericArray::from([1])
    }
}
impl<Size: typenum::Unsigned> MinSizeShape for (Cst<Size>,)
{
    type Value = Size;
}
impl MinSizeShape for (Dyn,) {
    type Value = typenum::U1;
}
impl<L: Dim + BroadcastShape<R, Output = O>, R: Dim, O: Dim> BroadcastShape<(R,)> for (L,) {
    type Output = (O,);
    #[inline(always)]
    fn broadcast(self, rhs: (R,)) -> Self::Output {
        (self.0.broadcast(rhs.0),)
    }
}
impl<L: Dim> BroadcastShape<()> for (L,) {
    type Output = (L,);
    #[inline(always)]
    fn broadcast(self, _: ()) -> Self::Output {
        self
    }
}

// Matrices
impl<D0: Dim, D1: Dim> Shape for (D0, D1,) {
    type Dims = typenum::U2;
    #[inline(always)]
    fn size(&self) -> usize {
        self.0.size() * self.1.size()
    }
    #[inline(always)]
    fn dimensions(&self) -> generic_array::GenericArray<usize, Self::Dims> {
        generic_array::GenericArray::from([self.0.size(), self.1.size()])
    }
    #[inline(always)]
    fn strides(&self) -> generic_array::GenericArray<usize, Self::Dims> {
        generic_array::GenericArray::from([self.1.size(), 1])
    }
}
impl<
    D0: Dim + MinSizeShape,
    D1: Dim + MinSizeShape,
    Size: typenum::Unsigned,
> MinSizeShape for (D0, D1,)
    where
        D0::Value: core::ops::Mul<D1::Value, Output = Size>,
{
    type Value = Size;
}
impl<
    D0: Dim + BroadcastShape<R0, Output = O0>,
    D1: Dim + BroadcastShape<R1, Output = O1>,
    R0: Dim,
    R1: Dim,
    O0: Dim,
    O1: Dim,
> BroadcastShape<(R0, R1,)> for (D0, D1,)
    where
        (O0, O1,): Shape,
{
    type Output = (O0, O1,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0, R1,)) -> Self::Output {
        (self.0.broadcast(rhs.0), self.1.broadcast(rhs.1))
    }
}
impl<
    D0: Dim,
    D1: Dim + BroadcastShape<R0, Output = O0>,
    R0: Dim,
    O0: Dim,
> BroadcastShape<(R0,)> for (D0, D1,)
    where
        (D0, O0,): Shape,
{
    type Output = (D0, O0,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0,)) -> Self::Output {
        (self.0, self.1.broadcast(rhs.0))
    }
}
impl<
    D0: Dim + BroadcastShape<R1, Output = O0>,
    R0: Dim,
    R1: Dim,
    O0: Dim,
> BroadcastShape<(R0, R1,)> for (D0,)
    where
        (R0, O0,): Shape,
{
    type Output = (R0, O0,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0, R1,)) -> Self::Output {
        (rhs.0, self.0.broadcast(rhs.1))
    }
}

// 3D tensors
impl<D0: Dim, D1: Dim, D2: Dim> Shape for (D0, D1, D2,) {
    type Dims = typenum::U3;
    #[inline(always)]
    fn size(&self) -> usize {
        self.0.size() * self.1.size() * self.2.size()
    }
    #[inline(always)]
    fn dimensions(&self) -> generic_array::GenericArray<usize, Self::Dims> {
        generic_array::GenericArray::from([self.0.size(), self.1.size(), self.2.size()])
    }
    #[inline(always)]
    fn strides(&self) -> generic_array::GenericArray<usize, Self::Dims> {
        generic_array::GenericArray::from([self.1.size() * self.2.size(), self.2.size(), 1])
    }
}
impl<
    D0: Dim + MinSizeShape,
    D1: Dim + MinSizeShape,
    D2: Dim + MinSizeShape,
    Size0: typenum::Unsigned,
    Size1: typenum::Unsigned,
> MinSizeShape for (D0, D1, D2,)
    where
        D0::Value: core::ops::Mul<D1::Value, Output = Size0>,
        Size0: core::ops::Mul<D2::Value, Output = Size1>,
{
    type Value = Size1;
}
impl<
    D0: Dim + BroadcastShape<R0, Output = O0>,
    D1: Dim + BroadcastShape<R1, Output = O1>,
    D2: Dim + BroadcastShape<R2, Output = O2>,
    R0: Dim,
    R1: Dim,
    R2: Dim,
    O0: Dim,
    O1: Dim,
    O2: Dim,
> BroadcastShape<(R0, R1, R2,)> for (D0, D1, D2,)
    where
        (O0, O1, O2,): Shape,
{
    type Output = (O0, O1, O2,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0, R1, R2,)) -> Self::Output {
        (self.0.broadcast(rhs.0), self.1.broadcast(rhs.1), self.2.broadcast(rhs.2))
    }
}
impl<
    D0: Dim,
    D1: Dim + BroadcastShape<R0, Output = O1>,
    D2: Dim + BroadcastShape<R1, Output = O2>,
    R0: Dim,
    R1: Dim,
    O1: Dim,
    O2: Dim,
> BroadcastShape<(R0, R1,)> for (D0, D1, D2,)
    where
        (D0, O1, O2,): Shape,
{
    type Output = (D0, O1, O2,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0, R1,)) -> Self::Output {
        (self.0, self.1.broadcast(rhs.0), self.2.broadcast(rhs.1))
    }
}
impl<
    D0: Dim,
    D1: Dim,
    D2: Dim + BroadcastShape<R0, Output = O2>,
    R0: Dim,
    O2: Dim,
> BroadcastShape<(R0,)> for (D0, D1, D2,)
    where
        (D0, D1, O2,): Shape,
{
    type Output = (D0, D1, O2,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0,)) -> Self::Output {
        (self.0, self.1, self.2.broadcast(rhs.0))
    }
}

// 4D tensors
impl<D0: Dim, D1: Dim, D2: Dim, D3: Dim> Shape for (D0, D1, D2, D3,) {
    type Dims = typenum::U4;
    #[inline(always)]
    fn size(&self) -> usize {
        self.0.size() * self.1.size() * self.2.size() * self.3.size()
    }
    #[inline(always)]
    fn dimensions(&self) -> generic_array::GenericArray<usize, Self::Dims> {
        generic_array::GenericArray::from([
            self.0.size(),
            self.1.size(),
            self.2.size(),
            self.3.size(),
        ])
    }
    #[inline(always)]
    fn strides(&self) -> generic_array::GenericArray<usize, Self::Dims> {
        generic_array::GenericArray::from([
            self.1.size() * self.2.size() * self.3.size(),
            self.2.size() * self.3.size(),
            self.3.size(),
            1,
        ])
    }
}
impl<
    D0: Dim + MinSizeShape,
    D1: Dim + MinSizeShape,
    D2: Dim + MinSizeShape,
    D3: Dim + MinSizeShape,
    Size0: typenum::Unsigned,
    Size1: typenum::Unsigned,
    Size2: typenum::Unsigned,
> MinSizeShape for (D0, D1, D2, D3,)
    where
        D0::Value: core::ops::Mul<D1::Value, Output = Size0>,
        Size0: core::ops::Mul<D2::Value, Output = Size1>,
        Size1: core::ops::Mul<D3::Value, Output = Size2>,
{
    type Value = Size2;
}
impl<
    D0: Dim + BroadcastShape<R0, Output = O0>,
    D1: Dim + BroadcastShape<R1, Output = O1>,
    D2: Dim + BroadcastShape<R2, Output = O2>,
    D3: Dim + BroadcastShape<R3, Output = O3>,
    R0: Dim,
    R1: Dim,
    R2: Dim,
    R3: Dim,
    O0: Dim,
    O1: Dim,
    O2: Dim,
    O3: Dim,
> BroadcastShape<(R0, R1, R2, R3,)> for (D0, D1, D2, D3,)
    where
        (O0, O1, O2, O3,): Shape,
{
    type Output = (O0, O1, O2, O3,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0, R1, R2, R3,)) -> Self::Output {
        (
            self.0.broadcast(rhs.0),
            self.1.broadcast(rhs.1),
            self.2.broadcast(rhs.2),
            self.3.broadcast(rhs.3),
        )
    }
}
impl<
    D0: Dim,
    D1: Dim + BroadcastShape<R0, Output = O1>,
    D2: Dim + BroadcastShape<R1, Output = O2>,
    D3: Dim + BroadcastShape<R2, Output = O3>,
    R0: Dim,
    R1: Dim,
    R2: Dim,
    O1: Dim,
    O2: Dim,
    O3: Dim,
> BroadcastShape<(R0, R1, R2,)> for (D0, D1, D2, D3,)
    where
        (D0, O1, O2, O3,): Shape,
{
    type Output = (D0, O1, O2, O3,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0, R1, R2,)) -> Self::Output {
        (
            self.0,
            self.1.broadcast(rhs.0),
            self.2.broadcast(rhs.1),
            self.3.broadcast(rhs.2),
        )
    }
}
impl<
    D0: Dim,
    D1: Dim,
    D2: Dim + BroadcastShape<R0, Output = O2>,
    D3: Dim + BroadcastShape<R1, Output = O3>,
    R0: Dim,
    R1: Dim,
    O2: Dim,
    O3: Dim,
> BroadcastShape<(R0, R1,)> for (D0, D1, D2, D3,)
    where
        (D0, D1, O2, O3,): Shape,
{
    type Output = (D0, D1, O2, O3,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0, R1,)) -> Self::Output {
        (self.0, self.1, self.2.broadcast(rhs.0), self.3.broadcast(rhs.1))
    }
}
impl<
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim + BroadcastShape<R0, Output = O3>,
    R0: Dim,
    O3: Dim,
> BroadcastShape<(R0,)> for (D0, D1, D2, D3,)
    where
        (D0, D1, D2, O3,): Shape,
{
    type Output = (D0, D1, D2, O3,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0,)) -> Self::Output {
        (self.0, self.1, self.2, self.3.broadcast(rhs.0))
    }
}

// 5D tensors
impl<D0: Dim, D1: Dim, D2: Dim, D3: Dim, D4: Dim> Shape for (D0, D1, D2, D3, D4,) {
    type Dims = typenum::U5;
    #[inline(always)]
    fn size(&self) -> usize {
        self.0.size() * self.1.size() * self.2.size() * self.3.size() * self.4.size()
    }
    #[inline(always)]
    fn dimensions(&self) -> generic_array::GenericArray<usize, Self::Dims> {
        generic_array::GenericArray::from([
            self.0.size(),
            self.1.size(),
            self.2.size(),
            self.3.size(),
            self.4.size(),
        ])
    }
    #[inline(always)]
    fn strides(&self) -> generic_array::GenericArray<usize, Self::Dims> {
        generic_array::GenericArray::from([
            self.1.size() * self.2.size() * self.3.size() * self.4.size(),
            self.2.size() * self.3.size() * self.4.size(),
            self.3.size() * self.4.size(),
            self.4.size(),
            1,
        ])
    }
}
impl<
    D0: Dim + MinSizeShape,
    D1: Dim + MinSizeShape,
    D2: Dim + MinSizeShape,
    D3: Dim + MinSizeShape,
    D4: Dim + MinSizeShape,
    Size0: typenum::Unsigned,
    Size1: typenum::Unsigned,
    Size2: typenum::Unsigned,
    Size3: typenum::Unsigned,
> MinSizeShape for (D0, D1, D2, D3, D4,)
    where
        D0::Value: core::ops::Mul<D1::Value, Output = Size0>,
        Size0: core::ops::Mul<D2::Value, Output = Size1>,
        Size1: core::ops::Mul<D3::Value, Output = Size2>,
        Size2: core::ops::Mul<D4::Value, Output = Size3>,
{
    type Value = Size3;
}
impl<
    D0: Dim + BroadcastShape<R0, Output = O0>,
    D1: Dim + BroadcastShape<R1, Output = O1>,
    D2: Dim + BroadcastShape<R2, Output = O2>,
    D3: Dim + BroadcastShape<R3, Output = O3>,
    D4: Dim + BroadcastShape<R4, Output = O4>,
    R0: Dim,
    R1: Dim,
    R2: Dim,
    R3: Dim,
    R4: Dim,
    O0: Dim,
    O1: Dim,
    O2: Dim,
    O3: Dim,
    O4: Dim,
> BroadcastShape<(R0, R1, R2, R3, R4,)> for (D0, D1, D2, D3, D4,)
    where
        (O0, O1, O2, O3, O4,): Shape,
{
    type Output = (O0, O1, O2, O3, O4,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0, R1, R2, R3, R4,)) -> Self::Output {
        (
            self.0.broadcast(rhs.0),
            self.1.broadcast(rhs.1),
            self.2.broadcast(rhs.2),
            self.3.broadcast(rhs.3),
            self.4.broadcast(rhs.4),
        )
    }
}
impl<
    D0: Dim,
    D1: Dim + BroadcastShape<R0, Output = O1>,
    D2: Dim + BroadcastShape<R1, Output = O2>,
    D3: Dim + BroadcastShape<R2, Output = O3>,
    D4: Dim + BroadcastShape<R3, Output = O4>,
    R0: Dim,
    R1: Dim,
    R2: Dim,
    R3: Dim,
    O1: Dim,
    O2: Dim,
    O3: Dim,
    O4: Dim,
> BroadcastShape<(R0, R1, R2, R3,)> for (D0, D1, D2, D3, D4,)
    where
        (D0, O1, O2, O3, O4,): Shape,
{
    type Output = (D0, O1, O2, O3, O4,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0, R1, R2, R3,)) -> Self::Output {
        (
            self.0,
            self.1.broadcast(rhs.0),
            self.2.broadcast(rhs.1),
            self.3.broadcast(rhs.2),
            self.4.broadcast(rhs.3),
        )
    }
}
impl<
    D0: Dim,
    D1: Dim,
    D2: Dim + BroadcastShape<R0, Output = O2>,
    D3: Dim + BroadcastShape<R1, Output = O3>,
    D4: Dim + BroadcastShape<R2, Output = O4>,
    R0: Dim,
    R1: Dim,
    R2: Dim,
    O2: Dim,
    O3: Dim,
    O4: Dim,
> BroadcastShape<(R0, R1, R2,)> for (D0, D1, D2, D3, D4,)
    where
        (D0, D1, O2, O3, O4,): Shape,
{
    type Output = (D0, D1, O2, O3, O4,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0, R1, R2,)) -> Self::Output {
        (
            self.0,
            self.1,
            self.2.broadcast(rhs.0),
            self.3.broadcast(rhs.1),
            self.4.broadcast(rhs.2),
        )
    }
}
impl<
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim + BroadcastShape<R0, Output = O3>,
    D4: Dim + BroadcastShape<R1, Output = O4>,
    R0: Dim,
    R1: Dim,
    O3: Dim,
    O4: Dim,
> BroadcastShape<(R0, R1,)> for (D0, D1, D2, D3, D4,)
    where
        (D0, D1, D2, O3, O4,): Shape,
{
    type Output = (D0, D1, D2, O3, O4,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0, R1,)) -> Self::Output {
        (
            self.0,
            self.1,
            self.2,
            self.3.broadcast(rhs.0),
            self.4.broadcast(rhs.1),
        )
    }
}
impl<
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D4: Dim + BroadcastShape<R0, Output = O4>,
    R0: Dim,
    O4: Dim,
> BroadcastShape<(R0,)> for (D0, D1, D2, D3, D4,)
    where
        (D0, D1, D2, D3, O4,): Shape,
{
    type Output = (D0, D1, D2, D3, O4,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0,)) -> Self::Output {
        (
            self.0,
            self.1,
            self.2,
            self.3,
            self.4.broadcast(rhs.0),
        )
    }
}

// 6D tensors
impl<D0: Dim, D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim> Shape for (D0, D1, D2, D3, D4, D5,) {
    type Dims = typenum::U6;
    #[inline(always)]
    fn size(&self) -> usize {
        self.0.size() * self.1.size() * self.2.size() * self.3.size() * self.4.size() * self.5.size()
    }
    #[inline(always)]
    fn dimensions(&self) -> generic_array::GenericArray<usize, Self::Dims> {
        generic_array::GenericArray::from([
            self.0.size(),
            self.1.size(),
            self.2.size(),
            self.3.size(),
            self.4.size(),
            self.5.size(),
        ])
    }
    #[inline(always)]
    fn strides(&self) -> generic_array::GenericArray<usize, Self::Dims> {
        generic_array::GenericArray::from([
            self.1.size() * self.2.size() * self.3.size() * self.4.size() * self.5.size(),
            self.2.size() * self.3.size() * self.4.size() * self.5.size(),
            self.3.size() * self.4.size() * self.5.size(),
            self.4.size() * self.5.size(),
            self.5.size(),
            1,
        ])
    }
}
impl<
    D0: Dim + MinSizeShape,
    D1: Dim + MinSizeShape,
    D2: Dim + MinSizeShape,
    D3: Dim + MinSizeShape,
    D4: Dim + MinSizeShape,
    D5: Dim + MinSizeShape,
    Size0: typenum::Unsigned,
    Size1: typenum::Unsigned,
    Size2: typenum::Unsigned,
    Size3: typenum::Unsigned,
    Size4: typenum::Unsigned,
> MinSizeShape for (D0, D1, D2, D3, D4, D5,)
    where
        D0::Value: core::ops::Mul<D1::Value, Output = Size0>,
        Size0: core::ops::Mul<D2::Value, Output = Size1>,
        Size1: core::ops::Mul<D3::Value, Output = Size2>,
        Size2: core::ops::Mul<D4::Value, Output = Size3>,
        Size3: core::ops::Mul<D5::Value, Output = Size4>,
{
    type Value = Size4;
}
impl<
    D0: Dim + BroadcastShape<R0, Output = O0>,
    D1: Dim + BroadcastShape<R1, Output = O1>,
    D2: Dim + BroadcastShape<R2, Output = O2>,
    D3: Dim + BroadcastShape<R3, Output = O3>,
    D4: Dim + BroadcastShape<R4, Output = O4>,
    D5: Dim + BroadcastShape<R5, Output = O5>,
    R0: Dim,
    R1: Dim,
    R2: Dim,
    R3: Dim,
    R4: Dim,
    R5: Dim,
    O0: Dim,
    O1: Dim,
    O2: Dim,
    O3: Dim,
    O4: Dim,
    O5: Dim,
> BroadcastShape<(R0, R1, R2, R3, R4, R5,)> for (D0, D1, D2, D3, D4, D5,)
    where
        (O0, O1, O2, O3, O4, O5,): Shape,
{
    type Output = (O0, O1, O2, O3, O4, O5,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0, R1, R2, R3, R4, R5,)) -> Self::Output {
        (
            self.0.broadcast(rhs.0),
            self.1.broadcast(rhs.1),
            self.2.broadcast(rhs.2),
            self.3.broadcast(rhs.3),
            self.4.broadcast(rhs.4),
            self.5.broadcast(rhs.5),
        )
    }
}
impl<
    D0: Dim,
    D1: Dim + BroadcastShape<R0, Output = O1>,
    D2: Dim + BroadcastShape<R1, Output = O2>,
    D3: Dim + BroadcastShape<R2, Output = O3>,
    D4: Dim + BroadcastShape<R3, Output = O4>,
    D5: Dim + BroadcastShape<R4, Output = O5>,
    R0: Dim,
    R1: Dim,
    R2: Dim,
    R3: Dim,
    R4: Dim,
    O1: Dim,
    O2: Dim,
    O3: Dim,
    O4: Dim,
    O5: Dim,
> BroadcastShape<(R0, R1, R2, R3, R4,)> for (D0, D1, D2, D3, D4, D5,)
    where
        (D0, O1, O2, O3, O4, O5,): Shape,
{
    type Output = (D0, O1, O2, O3, O4, O5,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0, R1, R2, R3, R4,)) -> Self::Output {
        (
            self.0,
            self.1.broadcast(rhs.0),
            self.2.broadcast(rhs.1),
            self.3.broadcast(rhs.2),
            self.4.broadcast(rhs.3),
            self.5.broadcast(rhs.4),
        )
    }
}
impl<
    D0: Dim,
    D1: Dim,
    D2: Dim + BroadcastShape<R0, Output = O2>,
    D3: Dim + BroadcastShape<R1, Output = O3>,
    D4: Dim + BroadcastShape<R2, Output = O4>,
    D5: Dim + BroadcastShape<R3, Output = O5>,
    R0: Dim,
    R1: Dim,
    R2: Dim,
    R3: Dim,
    O2: Dim,
    O3: Dim,
    O4: Dim,
    O5: Dim,
> BroadcastShape<(R0, R1, R2, R3,)> for (D0, D1, D2, D3, D4, D5,)
    where
        (D0, D1, O2, O3, O4, O5,): Shape,
{
    type Output = (D0, D1, O2, O3, O4, O5,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0, R1, R2, R3,)) -> Self::Output {
        (
            self.0,
            self.1,
            self.2.broadcast(rhs.0),
            self.3.broadcast(rhs.1),
            self.4.broadcast(rhs.2),
            self.5.broadcast(rhs.3),
        )
    }
}
impl<
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim + BroadcastShape<R0, Output = O3>,
    D4: Dim + BroadcastShape<R1, Output = O4>,
    D5: Dim + BroadcastShape<R2, Output = O5>,
    R0: Dim,
    R1: Dim,
    R2: Dim,
    O3: Dim,
    O4: Dim,
    O5: Dim,
> BroadcastShape<(R0, R1, R2,)> for (D0, D1, D2, D3, D4, D5,)
    where
        (D0, D1, D2, O3, O4, O5,): Shape,
{
    type Output = (D0, D1, D2, O3, O4, O5,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0, R1, R2,)) -> Self::Output {
        (
            self.0,
            self.1,
            self.2,
            self.3.broadcast(rhs.0),
            self.4.broadcast(rhs.1),
            self.5.broadcast(rhs.2),
        )
    }
}
impl<
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D4: Dim + BroadcastShape<R0, Output = O4>,
    D5: Dim + BroadcastShape<R1, Output = O5>,
    R0: Dim,
    R1: Dim,
    O4: Dim,
    O5: Dim,
> BroadcastShape<(R0, R1,)> for (D0, D1, D2, D3, D4, D5,)
    where
        (D0, D1, D2, D3, O4, O5,): Shape,
{
    type Output = (D0, D1, D2, D3, O4, O5,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0, R1,)) -> Self::Output {
        (
            self.0,
            self.1,
            self.2,
            self.3,
            self.4.broadcast(rhs.0),
            self.5.broadcast(rhs.1),
        )
    }
}
impl<
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D4: Dim,
    D5: Dim + BroadcastShape<R0, Output = O5>,
    R0: Dim,
    O5: Dim,
> BroadcastShape<(R0,)> for (D0, D1, D2, D3, D4, D5,)
    where
        (D0, D1, D2, D3, D4, O5,): Shape,
{
    type Output = (D0, D1, D2, D3, D4, O5,);
    #[inline(always)]
    fn broadcast(self, rhs: (R0,)) -> Self::Output {
        (
            self.0,
            self.1,
            self.2,
            self.3,
            self.4,
            self.5.broadcast(rhs.0),
        )
    }
}