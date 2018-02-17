use crate::transform::{MirPass, MirSource};
use rustc::mir::visit::{MutVisitor, NonUseContext, PlaceContext, Visitor};
use rustc::mir::*;
use rustc::ty::layout::VariantIdx;
use rustc::ty::util::IntTypeExt;
use rustc::ty::{self, Ty, TyCtxt};
use rustc_index::vec::IndexVec;
use std::collections::BTreeMap;
use std::iter::Step;
use std::ops::Range;
use syntax_pos::Span;

pub struct SplitLocalFields;

impl MirPass<'tcx> for SplitLocalFields {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, _: MirSource<'tcx>, body: &mut BodyAndCache<'tcx>) {
        let mut collector = FieldTreeCollector {
            tcx,
            locals: body.local_decls.iter().map(|decl| FieldTreeNode::new(decl.ty)).collect(),
        };

        // Can't split return and arguments.
        collector.locals[RETURN_PLACE].make_opaque();
        for arg in body.args_iter() {
            collector.locals[arg].make_opaque();
        }

        collector.visit_body(read_only!(body));

        let replacements = collector
            .locals
            .iter_enumerated_mut()
            .map(|(local, root)| {
                // Don't rename locals that are entirely opaque.
                match root.kind {
                    FieldTreeNodeKind::Opaque { .. } => local..local.add_one(),
                    FieldTreeNodeKind::Split(_) => {
                        let source_info = body.local_decls[local].source_info;
                        let first = body.local_decls.next_index();
                        root.assign_locals(&mut body.local_decls, source_info);
                        first..body.local_decls.next_index()
                    }
                }
            })
            .collect::<IndexVec<Local, Range<Local>>>();

        // Expand `Storage{Live,Dead}` statements to refer to the replacement locals.
        for bb in body.basic_blocks_mut() {
            bb.expand_statements(|stmt| {
                let (local, is_live) = match stmt.kind {
                    StatementKind::StorageLive(local) => (local, true),
                    StatementKind::StorageDead(local) => (local, false),
                    _ => return None,
                };
                let range = replacements[local].clone();
                // FIXME(eddyb) `Range<Local>` should itself be iterable.
                let range = (range.start.as_u32()..range.end.as_u32()).map(Local::from_u32);
                let source_info = stmt.source_info;
                Some(range.map(move |new_local| Statement {
                    source_info,
                    kind: if is_live {
                        StatementKind::StorageLive(new_local)
                    } else {
                        StatementKind::StorageDead(new_local)
                    },
                }))
            });
        }
        drop(replacements);

        // Lastly, replace all the opaque nodes with their new locals.
        let mut replacer = FieldTreeReplacer { tcx, span: body.span, locals: collector.locals };
        replacer.visit_body(body);
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Fragment {
    Discriminant,
    Field(Option<VariantIdx>, Field),
}

struct FieldTreeNode<'tcx> {
    ty: Ty<'tcx>,
    kind: FieldTreeNodeKind<'tcx>,
}

enum FieldTreeNodeKind<'tcx> {
    /// This node needs to remain self-contained, e.g. due to accesses / borrows.
    Opaque { replacement_local: Option<Local> },

    /// This node can be split into separate locals for its fields.
    Split(BTreeMap<Fragment, FieldTreeNode<'tcx>>),
}

impl FieldTreeNode<'tcx> {
    fn new(ty: Ty<'tcx>) -> Self {
        let mut node = FieldTreeNode { ty, kind: FieldTreeNodeKind::Split(BTreeMap::new()) };

        if let ty::Adt(adt_def, _) = ty.kind {
            // Unions have (observably) overlapping members, so don't split them.
            if adt_def.is_union() {
                node.make_opaque();
            }
        }

        node
    }

    fn fragment(&mut self, fragment: Fragment, ty: Ty<'tcx>) -> Option<&mut Self> {
        match self.kind {
            FieldTreeNodeKind::Split(ref mut fragments) => {
                Some(fragments.entry(fragment).or_insert_with(|| FieldTreeNode::new(ty)))
            }
            FieldTreeNodeKind::Opaque { .. } => None,
        }
    }

    fn discriminant(&mut self, tcx: TyCtxt<'tcx>) -> Option<&mut Self> {
        match self.ty.kind {
            ty::Adt(adt_def, _) if adt_def.is_enum() => {
                let discr_ty = adt_def.repr.discr_type().to_ty(tcx);
                self.fragment(Fragment::Discriminant, discr_ty)
            }
            _ => None,
        }
    }

    fn make_opaque(&mut self) {
        if let FieldTreeNodeKind::Split(_) = self.kind {
            self.kind = FieldTreeNodeKind::Opaque { replacement_local: None };
        }
    }

    fn project(
        mut self: &'a mut Self,
        mut proj_elems: &'tcx [PlaceElem<'tcx>],
    ) -> (&'a mut Self, &'tcx [PlaceElem<'tcx>]) {
        let mut variant_index = None;
        while let [elem, rest @ ..] = proj_elems {
            if let FieldTreeNodeKind::Opaque { .. } = self.kind {
                break;
            }

            match *elem {
                ProjectionElem::Field(f, ty) => {
                    let field = Fragment::Field(variant_index, f);
                    // FIXME(eddyb) use `self.fragment(field)` post-Polonius(?).
                    match self.kind {
                        FieldTreeNodeKind::Split(ref mut fragments) => {
                            self = fragments.entry(field).or_insert_with(|| FieldTreeNode::new(ty));
                        }
                        FieldTreeNodeKind::Opaque { .. } => unreachable!(),
                    }
                }

                ProjectionElem::Downcast(..) => {}

                // FIXME(eddyb) support indexing by constants.
                ProjectionElem::ConstantIndex { .. } | ProjectionElem::Subslice { .. } |
                // Can't support without alias analysis.
                ProjectionElem::Index(_) | ProjectionElem::Deref => {
                    // If we can't project, we must be opaque.
                    self.make_opaque();
                    break;
                }
            }

            proj_elems = rest;
            variant_index = match *elem {
                ProjectionElem::Downcast(_, v) => Some(v),
                _ => None,
            };
        }

        (self, proj_elems)
    }

    fn assign_locals(
        &mut self,
        local_decls: &mut IndexVec<Local, LocalDecl<'tcx>>,
        source_info: SourceInfo,
    ) {
        match self.kind {
            FieldTreeNodeKind::Opaque { ref mut replacement_local } => {
                let mut decl = LocalDecl::new_internal(self.ty, source_info.span);
                decl.source_info = source_info;
                *replacement_local = Some(local_decls.push(decl));
            }
            FieldTreeNodeKind::Split(ref mut fragments) => {
                for fragment in fragments.values_mut() {
                    fragment.assign_locals(local_decls, source_info);
                }
            }
        }
    }
}

struct FieldTreeCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    locals: IndexVec<Local, FieldTreeNode<'tcx>>,
}

impl FieldTreeCollector<'tcx> {
    fn place_node(&'a mut self, place: &Place<'tcx>) -> Option<&'a mut FieldTreeNode<'tcx>> {
        let base_local = match place.base {
            PlaceBase::Local(local) => local,
            PlaceBase::Static(_) => return None,
        };
        let (node, proj_elems) = self.locals[base_local].project(place.projection);
        if proj_elems.is_empty() { Some(node) } else { None }
    }
}

impl Visitor<'tcx> for FieldTreeCollector<'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _: Location) {
        if let Some(node) = self.place_node(place) {
            if context.is_use() {
                node.make_opaque();
            }

            // FIXME(eddyb) implement debuginfo support for split locals.
            if let PlaceContext::NonUse(NonUseContext::VarDebugInfo) = context {
                node.make_opaque();
            }
        }
    }

    // Special-case `(Set)Discriminant(place)` to only mark `Fragment::Discriminant` as opaque.
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        let tcx = self.tcx;

        if let Rvalue::Discriminant(ref place) = *rvalue {
            if let Some(node) = self.place_node(place) {
                if let Some(discr) = node.discriminant(tcx) {
                    discr.make_opaque();
                }
            }
        } else {
            self.super_rvalue(rvalue, location);
        }
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        let tcx = self.tcx;

        if let StatementKind::SetDiscriminant { ref place, .. } = statement.kind {
            if let Some(node) = self.place_node(place) {
                if let Some(discr) = node.discriminant(tcx) {
                    discr.make_opaque();
                }
            }
        } else {
            self.super_statement(statement, location);
        }
    }
}

struct FieldTreeReplacer<'tcx> {
    tcx: TyCtxt<'tcx>,
    span: Span,
    locals: IndexVec<Local, FieldTreeNode<'tcx>>,
}

impl FieldTreeReplacer<'tcx> {
    fn replace(
        &mut self,
        place: &Place<'tcx>,
    ) -> Option<Result<Place<'tcx>, &mut FieldTreeNode<'tcx>>> {
        let base_local = match place.base {
            PlaceBase::Local(local) => local,
            PlaceBase::Static(_) => return None,
        };
        let base_node = &mut self.locals[base_local];

        // Avoid identity replacements, which would re-intern projections.
        if let FieldTreeNodeKind::Opaque { replacement_local: None } = base_node.kind {
            return None;
        }

        let (node, proj_elems) = base_node.project(place.projection);

        Some(match node.kind {
            FieldTreeNodeKind::Opaque { replacement_local } => Ok(Place {
                base: PlaceBase::Local(replacement_local.expect("missing replacement")),
                projection: self.tcx.intern_place_elems(proj_elems),
            }),

            // HACK(eddyb) this only exists to support `(Set)Discriminant` below.
            FieldTreeNodeKind::Split(_) => {
                assert_eq!(proj_elems, &[]);

                Err(node)
            }
        })
    }
}

impl MutVisitor<'tcx> for FieldTreeReplacer<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, _: PlaceContext, _: Location) {
        if let Some(place_replacement) = self.replace(place) {
            match place_replacement {
                Ok(place_replacement) => *place = place_replacement,
                // HACK(eddyb) this only exists to support `(Set)Discriminant` below.
                Err(_) => unreachable!(),
            }
        }
    }

    // Special-case `(Set)Discriminant(place)` to use `discr_local` for `place`.
    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>, location: Location) {
        let tcx = self.tcx;
        let span = self.span;

        if let Rvalue::Discriminant(ref mut place) = rvalue {
            if let Some(place_replacement) = self.replace(place) {
                match place_replacement {
                    Ok(place_replacement) => *place = place_replacement,
                    Err(node) => {
                        let discr = if let Some(discr) = node.discriminant(tcx) {
                            let discr = match discr.kind {
                                FieldTreeNodeKind::Opaque { replacement_local } => {
                                    replacement_local
                                }
                                FieldTreeNodeKind::Split(_) => unreachable!(),
                            };
                            Operand::Copy(Place::from(discr.expect("missing discriminant")))
                        } else {
                            // Non-enums don't have discriminants other than `0u8`.
                            let discr_value = ty::Const::from_bits(
                                tcx,
                                0,
                                ty::ParamEnv::empty().and(tcx.types.u8),
                            );
                            Operand::Constant(box Constant {
                                span,
                                user_ty: None,
                                literal: discr_value,
                            })
                        };
                        *rvalue = Rvalue::Use(discr);
                    }
                }
            }
        } else {
            self.super_rvalue(rvalue, location);
        }
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        self.span = statement.source_info.span;

        let tcx = self.tcx;
        let span = self.span;

        if let StatementKind::SetDiscriminant { ref mut place, variant_index } = statement.kind {
            if let Some(place_replacement) = self.replace(place) {
                match place_replacement {
                    Ok(place_replacement) => **place = place_replacement,
                    Err(node) => {
                        if let Some(discr) = node.discriminant(tcx) {
                            let discr_ty = discr.ty;
                            let discr_local = match discr.kind {
                                FieldTreeNodeKind::Opaque { replacement_local } => {
                                    replacement_local
                                }
                                FieldTreeNodeKind::Split(_) => unreachable!(),
                            };
                            let discr_place =
                                Place::from(discr_local.expect("missing discriminant"));
                            let discr_value = ty::Const::from_bits(
                                tcx,
                                node.ty.discriminant_for_variant(tcx, variant_index).unwrap().val,
                                ty::ParamEnv::empty().and(discr_ty),
                            );
                            let discr_rvalue = Rvalue::Use(Operand::Constant(box Constant {
                                span,
                                user_ty: None,
                                literal: discr_value,
                            }));
                            statement.kind = StatementKind::Assign(box (discr_place, discr_rvalue));
                        } else {
                            // Non-enums don't have discriminants to set.
                            statement.kind = StatementKind::Nop;
                        }
                    }
                }
            }
        } else {
            self.super_statement(statement, location);
        }
    }
}
